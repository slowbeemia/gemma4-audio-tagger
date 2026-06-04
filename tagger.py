#!/usr/bin/env python3
"""
gemma4-audio-tagger — Batch audio tagging via Gemma 4 12B (llama.cpp).

Connects to llama-server on localhost:8001. Audio is base64-encoded 16kHz
mono WAV, sent as input_audio content blocks via the OpenAI-compatible API.

Usage:
    uv run python3 tagger.py                  # tag next 10 untagged
    uv run python3 tagger.py --max 10         # tag 10 files
    uv run python3 tagger.py --max 1          # test one file
    uv run python3 tagger.py --dry-run        # show what would be tagged
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# --- Config ----------------------------------------------------------------

_ENV_FILE = Path(__file__).parent / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8001")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "gemma-4-12b-it")
ASSET_MANAGER_URL = os.getenv("ASSET_MANAGER_URL", "http://bob.fritz.box:8899")
DELAY_SEC = float(os.getenv("DELAY_SEC", "2"))
API_PAGE_SIZE = int(os.getenv("API_PAGE_SIZE", "50"))
PATH_MAP_FROM = os.getenv("PATH_MAP_FROM", "//orange/Assets")
PATH_MAP_TO = os.getenv("PATH_MAP_TO", "/mnt/orange")

STATE_FILE = Path(__file__).parent / ".tagger_state.json"
LOG_FILE = Path(__file__).parent / "tagger.log"

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


def fetch_untagged(client: httpx.Client, page_size: int = 50) -> list[dict]:
    all_assets = []
    offset = 0
    while True:
        try:
            resp = client.get(
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


def audio_to_base64_wav(filepath: Path) -> str | None:
    """Convert any audio file to 16kHz mono WAV, return base64 string."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(filepath),
            "-ar", "16000", "-ac", "1",
            "-loglevel", "error", tmp_path,
        ]
        subprocess.run(cmd, check=True, timeout=10)

        with open(tmp_path, "rb") as f:
            wav_bytes = f.read()

        os.unlink(tmp_path)
        return base64.b64encode(wav_bytes).decode("ascii")
    except Exception as e:
        log(f"  ERROR: ffmpeg conversion failed for {filepath}: {e}")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None


def tag_audio(full_path: str, filename: str, rel_path: str) -> dict | None:
    """Send audio to llama-server via input_audio content block."""
    local_path = full_path.replace(PATH_MAP_FROM, PATH_MAP_TO, 1)
    path = Path(local_path)
    if not path.exists():
        log(f"  SKIP: file not found: {local_path}")
        return None

    b64_wav = audio_to_base64_wav(path)
    if not b64_wav:
        return None

    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "input_audio", "input_audio": {"data": b64_wav}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(120)) as client:
            resp = client.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
    except httpx.HTTPError as e:
        log(f"  ERROR: llama-server request failed: {e}")
        return None

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # Log raw output for debugging
    log(f"  [RAW] {repr(content)[:300]}")

    # Clean markdown code fences
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                log(f"  ERROR: JSON parse failed on regex match: {e}")
        log(f"  ERROR: Content is not valid JSON.")
        return None


def send_to_asset_manager(asset_id: int, tags: dict, client: httpx.Client) -> bool:
    try:
        resp = client.post(
            f"{ASSET_MANAGER_URL}/api/audio-assets/{asset_id}/tag",
            json={
                "tags": tags.get("tags", ""),
                "description": tags.get("description", ""),
                "notes": "gemma4-12b-llama.cpp",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("ok", False)
    except Exception as e:
        log(f"  ERROR posting to Asset Manager: {e}")
        return False


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"tagged": 0, "errors": 0, "processed": [], "total": 0, "runtime_s": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 12B Audio Tagger (llama.cpp)")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    log("=" * 60)
    log("Gemma 4 12B Audio Tagger (llama.cpp)")
    log(f"Server: {LLAMA_SERVER_URL}")
    log(f"Model:  {LLAMA_MODEL}")
    log(f"Asset Manager: {ASSET_MANAGER_URL}")

    # Quick health check
    try:
        r = httpx.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=5)
        r.raise_for_status()
        log(f"llama-server: OK ({r.json().get('data', [{}])[0].get('id', '?')})")
    except Exception as e:
        log(f"FATAL: llama-server not reachable: {e}")
        sys.exit(1)

    mgr = httpx.Client(timeout=httpx.Timeout(30))

    log("Fetching untagged audio from Asset Manager...")
    assets = fetch_untagged(mgr, API_PAGE_SIZE)
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
    max_count = args.max if args.max else 10
    state["total"] = max_count
    start_time = time.time()

    tagged = 0
    for asset in assets:
        asset_id = asset["id"]
        filename = asset["filename"]
        full_path = asset["full_path"]

        if str(asset_id) in processed:
            continue
        if tagged >= max_count:
            break

        log(f"[{tagged + 1}/{max_count}] #{asset_id} {filename}")

        result = tag_audio(full_path, filename, asset.get("rel_path", filename))
        if result:
            log(f"  Tags: {result.get('tags', '—')[:100]}")
            if result.get("description"):
                log(f"  Desc:  {result['description'][:120]}")
            if send_to_asset_manager(asset_id, result, mgr):
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
