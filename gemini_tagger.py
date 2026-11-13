#!/usr/bin/env python3
"""
gemini_tagger.py — Batch audio tagging via Gemini 3.5 Flash (Google Cloud).

Runs on Bob (no GPU needed). Uses the same audio interface as Gemma 4 12B —
Google Gemini models natively understand audio via the Files API.

Pipeline:
    1. Fetch untagged audio assets from Asset Manager API (:8899)
    2. Upload each to Gemini Files API
    3. Generate tags + German description via gemini-3.5-flash
    4. Post results back to Asset Manager

Usage:
    uv run python3 gemini_tagger.py                  # tag next 10 untagged
    uv run python3 gemini_tagger.py --max 5          # tag 5 files
    uv run python3 gemini_tagger.py --all            # all untagged (watch limits!)
    uv run python3 gemini_tagger.py --asset-id 244   # single asset
    uv run python3 gemini_tagger.py --dry-run        # show what would be tagged
    uv run python3 gemini_tagger.py --status         # show daily limit status
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Load env
_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
# Also try Hermes .env for GOOGLE_API_KEY
_HERMES_ENV = Path.home() / ".hermes" / ".env"
if _HERMES_ENV.exists() and not os.environ.get("GOOGLE_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv(_HERMES_ENV)

from google import genai
from google.genai import types

# --- Config ----------------------------------------------------------------

MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
ASSET_MANAGER_URL = os.getenv("ASSET_MANAGER_URL", "http://bob.fritz.box:8899")
DELAY_SEC = float(os.getenv("DELAY_SEC", "3"))
API_PAGE_SIZE = int(os.getenv("API_PAGE_SIZE", "50"))
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "1500"))  # Gemini free tier
PATH_MAP_FROM = os.getenv("PATH_MAP_FROM", "//orange/Assets")
PATH_MAP_TO = os.getenv("PATH_MAP_TO", "//orange/Assets")

STATE_FILE = Path(__file__).parent / ".gemini_tagger_state.json"
LOG_FILE = Path(__file__).parent / "gemini_tagger.log"

SYSTEM_PROMPT = """You are an audio analysis expert for game development. Listen carefully to the audio and respond ONLY with a JSON object:

{"tags": "tag1, tag2, tag3, tag4, tag5, tag6", "description": "Kurze deutsche Beschreibung des Sounds (1 Satz, game-dev-relevant)"}

Tags should be game-dev relevant categories (e.g. explosion, coin_loot, power_up, laser, ui_click, ambient_dark, magic_spell, footsteps, etc.).
Output ONLY the JSON, no other text."""

USER_PROMPT = "Analyze this audio clip for a game asset library. What kind of sound is this? Give 5-8 precise tags and a short German description."


def log(msg: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def fetch_untagged(client_http: httpx.Client, page_size: int = 50) -> list[dict]:
    all_assets = []
    offset = 0
    while True:
        try:
            resp = client_http.get(
                f"{ASSET_MANAGER_URL}/api/audio-assets/untagged",
                params={"limit": page_size, "offset": offset},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("assets", [])
            total = data.get("total", 0)
            all_assets.extend(batch)
            offset += page_size
            if offset >= total:
                break
        except httpx.HTTPError as e:
            log(f"ERROR fetching assets at offset {offset}: {e}")
            break
    return all_assets


def tag_one(client_gemini: genai.Client, local_path: Path) -> dict | None:
    """Upload audio to Gemini, get tags, cleanup. Returns dict or None."""
    if not local_path.exists():
        log(f"  SKIP: file not found: {local_path}")
        return None

    # Upload
    try:
        uploaded = client_gemini.files.upload(file=local_path)
    except Exception as e:
        log(f"  ERROR upload: {e}")
        return None

    # Wait for ACTIVE
    for _ in range(10):
        if uploaded.state.name != "PROCESSING":
            break
        time.sleep(1)
        uploaded = client_gemini.files.get(name=uploaded.name)

    if uploaded.state.name != "ACTIVE":
        log(f"  ERROR: file state={uploaded.state.name}")
        _cleanup(client_gemini, uploaded.name)
        return None

    # Generate
    try:
        response = client_gemini.models.generate_content(
            model=MODEL_ID,
            contents=[uploaded, USER_PROMPT],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                max_output_tokens=300,
            ),
        )
    except Exception as e:
        log(f"  ERROR generation: {e}")
        _cleanup(client_gemini, uploaded.name)
        return None

    _cleanup(client_gemini, uploaded.name)
    text = response.text.strip()

    # Parse JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        log(f"  ERROR: JSON parse failed. Raw: {text[:200]}")
        return None


def send_to_asset_manager(asset_id: int, tags: dict, client_http: httpx.Client) -> bool:
    try:
        resp = client_http.post(
            f"{ASSET_MANAGER_URL}/api/audio-assets/{asset_id}/tag",
            json={
                "tags": tags.get("tags", ""),
                "description": tags.get("description", ""),
                "notes": f"gemini-{MODEL_ID}",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("ok", False)
    except Exception as e:
        log(f"  ERROR posting to Asset Manager: {e}")
        return False


def _cleanup(client_gemini, file_name: str) -> None:
    try:
        client_gemini.files.delete(name=file_name)
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"tagged": 0, "errors": 0, "processed": [], "total": 0, "runtime_s": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def show_status() -> None:
    state = load_state()
    print(f"📊 Gemini Tagger Status")
    print(f"   Modell:      {MODEL_ID}")
    print(f"   Getaggt:     {state.get('tagged', 0)}")
    print(f"   Fehler:      {state.get('errors', 0)}")
    print(f"   Laufzeit:    {state.get('runtime_s', 0)}s")
    print(f"   Asset Mgr:   {ASSET_MANAGER_URL}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gemini Audio Tagger")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--asset-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--retag", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        log("FATAL: GOOGLE_API_KEY not set")
        sys.exit(1)

    log("=" * 60)
    log(f"Gemini Audio Tagger — Model: {MODEL_ID}")
    log(f"Asset Manager: {ASSET_MANAGER_URL}")

    client_gemini = genai.Client(api_key=api_key)
    client_http = httpx.Client(timeout=httpx.Timeout(30))

    # Single asset mode
    if args.asset_id:
        resp = client_http.get(
            f"{ASSET_MANAGER_URL}/api/audio-assets/{args.asset_id}",
            timeout=10,
        )
        if resp.status_code != 200:
            log(f"ERROR: Asset #{args.asset_id} not found")
            sys.exit(1)
        asset = resp.json()
        local_path = asset["full_path"].replace(PATH_MAP_FROM, PATH_MAP_TO, 1)
        log(f"Single: #{asset['id']} {asset['filename']}")
        result = tag_one(client_gemini, Path(local_path))
        if result:
            log(f"  Tags: {result.get('tags', '—')}")
            log(f"  Desc:  {result.get('description', '—')}")
            if send_to_asset_manager(asset["id"], result, client_http):
                log("  ✓ Posted to Asset Manager")
            else:
                log("  ✗ Failed to post")
        return

    # Batch mode
    log("Fetching untagged audio from Asset Manager...")
    assets = fetch_untagged(client_http, API_PAGE_SIZE)
    log(f"Found {len(assets)} untagged audio assets")

    if args.dry_run:
        for a in assets[:20]:
            print(f"  #{a['id']}  {a['filename']}  ({a.get('duration', 0):.1f}s)")
        if len(assets) > 20:
            print(f"  ... and {len(assets) - 20} more")
        return

    state = load_state()
    if args.reset:
        state = {"tagged": 0, "errors": 0, "processed": [], "total": 0, "runtime_s": 0}
    processed = set(state.get("processed", []))
    max_count = args.max if args.max else (len(assets) if args.all else 10)
    state["total"] = max_count
    start_time = time.time()

    tagged = 0
    for asset in assets:
        asset_id = asset["id"]
        filename = asset["filename"]
        full_path = asset["full_path"]

        if str(asset_id) in processed and not args.retag:
            continue
        if tagged >= max_count:
            break

        local_path = full_path.replace(PATH_MAP_FROM, PATH_MAP_TO, 1)
        log(f"[{tagged + 1}/{max_count}] #{asset_id} {filename}")

        result = tag_one(client_gemini, Path(local_path))
        if result:
            log(f"  Tags: {result.get('tags', '—')}")
            if result.get("description"):
                log(f"  Desc:  {result['description'][:120]}")
            if send_to_asset_manager(asset_id, result, client_http):
                tagged += 1
                state["tagged"] += 1
            else:
                state["errors"] += 1
        else:
            state["errors"] += 1

        processed.add(str(asset_id))
        state["processed"] = list(processed)
        state["runtime_s"] = int(time.time() - start_time)

        if tagged % 5 == 0:
            save_state(state)

        time.sleep(DELAY_SEC)

    state["runtime_s"] = int(time.time() - start_time)
    save_state(state)
    log(f"Done! Tagged: {tagged}, Errors: {state['errors']}, Runtime: {state['runtime_s']}s")


if __name__ == "__main__":
    main()
