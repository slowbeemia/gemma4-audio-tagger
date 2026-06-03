#!/usr/bin/env python3
"""
gemma4-audio-tagger — Batch audio tagging via Gemma 4 12B on llama.cpp.

Queries Asset Manager API for untagged audio files, processes them locally
via llama-server, and sends tags + descriptions back.

Usage:
    uv run python3 tagger.py                  # tag all untagged audio
    uv run python3 tagger.py --max 10         # tag 10 files, then stop
    uv run python3 tagger.py --dry-run        # show what would be tagged
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
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
ASSET_MANAGER_URL = os.getenv("ASSET_MANAGER_URL", "http://bob.fritz.box:8899")
DELAY_SEC = float(os.getenv("DELAY_SEC", "2"))
API_PAGE_SIZE = int(os.getenv("API_PAGE_SIZE", "50"))

STATE_FILE = Path(__file__).parent / ".tagger_state.json"
LOG_FILE = Path(__file__).parent / "tagger.log"

# --- Audio prompt (30s max for Gemma 4 12B) --------------------------------

SYSTEM_PROMPT = """You are an audio analysis expert for game development. Your task is to tag and describe short sound effect / music clips.

For each audio clip, output a JSON object with these fields:
- "tags": comma-separated string of 3-8 lowercase tags (e.g. "explosion, impact, boom, deep, cinematic")
- "description": short German description (1 sentence, practical, game-dev focused)

Output ONLY the JSON object, no other text."""

USER_PROMPT_TEMPLATE = """Tag this audio clip for a game asset library.

Audio file: {filename}
Path: {rel_path}

Output the JSON object now."""


def log(msg: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def convert_to_wav_16khz(audio_path: Path, max_duration: float = 0) -> bytes:
    """Convert any audio file to 16kHz mono WAV bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1",
            "-sample_fmt", "s16",
        ]
        if max_duration > 0:
            cmd += ["-t", str(max_duration)]
        cmd += ["-loglevel", "error", str(tmp_path)]

        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch_untagged_assets(client: httpx.Client, page_size: int = 50) -> list[dict]:
    """Fetch ALL untagged audio assets from Asset Manager API (paginated)."""
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
            print(f"  ... fetched {len(all_assets)}/{total} from Asset Manager", flush=True)
            offset += page_size
            if offset >= total:
                break
        except httpx.HTTPError as e:
            log(f"ERROR fetching assets at offset {offset}: {e}")
            break

    return all_assets


def tag_audio(full_path: str, filename: str, rel_path: str, client: httpx.Client) -> dict | None:
    """Send audio to llama-server, get tags back."""
    path = Path(full_path)
    if not path.exists():
        log(f"  SKIP: file not found: {full_path}")
        return None

    # Convert to 16kHz WAV — trim to 20s if longer
    try:
        wav_bytes = convert_to_wav_16khz(path, max_duration=20.0)
    except subprocess.CalledProcessError as e:
        log(f"  ERROR: ffmpeg conversion failed: {e.stderr.decode()[:200]}")
        return None

    # Log if trimmed
    wav_size = len(wav_bytes)
    audio_duration = (wav_size - 44) / 32000 if wav_size > 44 else 0
    if audio_duration > 19.9:
        log(f"  (original >20s, using first 20s)")

    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    # Send to llama-server
    payload = {
        "model": "gemma-4-12b-it",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT_TEMPLATE.format(filename=filename, rel_path=rel_path)},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                ],
            },
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 300,
    }

    try:
        resp = client.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        log(f"  ERROR: llama-server request failed: {e}")
        return None

    # Parse response
    try:
        content = data["choices"][0]["message"]["content"]
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        return json.loads(content)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        log(f"  ERROR: failed to parse response: {e}")
        log(f"  Raw: {content[:300]}")
        return None


def send_to_asset_manager(asset_id: int, tags: dict, client: httpx.Client) -> bool:
    """Send tags to Asset Manager API."""
    try:
        resp = client.post(
            f"{ASSET_MANAGER_URL}/api/audio-assets/{asset_id}/tag",
            json={
                "tags": tags.get("tags", ""),
                "description": tags.get("description", ""),
                "notes": "gemma4-12b",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log(f"  ERROR: Asset Manager API failed: {e}")
        return False


def report_status(client: httpx.Client, state: dict, status: str = "running") -> None:
    """Report progress to Asset Manager status endpoint."""
    try:
        client.post(
            f"{ASSET_MANAGER_URL}/api/tagging-status",
            json={
                "host": "thegrey-gemma4",
                "model": "gemma-4-12b-it",
                "version": "1.0",
                "status": status,
                "current": state["tagged"],
                "total": state["total"],
                "errors": state["errors"],
                "runtime_s": state.get("runtime_s", 0),
            },
            timeout=5,
        )
    except Exception as e:
        log(f"Status report failed: {e}")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"tagged": 0, "errors": 0, "processed": [], "total": 0, "runtime_s": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 12B Audio Tagger")
    parser.add_argument("--max", type=int, default=0, help="Max files to tag (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tagged (fetches list only)")
    parser.add_argument("--reset", action="store_true", help="Reset progress state")
    args = parser.parse_args()

    log("=" * 60)
    log("Gemma 4 12B Audio Tagger starting")
    log(f"Server: {LLAMA_SERVER_URL}")
    log(f"Asset Manager: {ASSET_MANAGER_URL}")

    # HTTP client for Asset Manager
    mgr = httpx.Client(timeout=httpx.Timeout(30))

    # Fetch untagged assets from Asset Manager DB
    log("Fetching untagged audio from Asset Manager...")
    assets = fetch_untagged_assets(mgr, API_PAGE_SIZE)

    if args.dry_run:
        log(f"Dry-run: {len(assets)} untagged audio assets would be tagged")
        for a in assets[:20]:
            print(f"  #{a['id']}  {a['filename']}  ({a.get('duration', 0):.1f}s)  {a['rel_path']}")
        if len(assets) > 20:
            print(f"  ... and {len(assets) - 20} more")
        return

    # Load state
    state = load_state()
    if args.reset:
        state = {"tagged": 0, "errors": 0, "processed": [], "total": 0, "runtime_s": 0}
    processed = set(state.get("processed", []))
    state["total"] = len(assets)
    start_time = time.time()

    # Health check llama-server
    try:
        resp = httpx.get(f"{LLAMA_SERVER_URL}/health", timeout=5)
        log(f"llama-server health: {resp.json()}")
    except Exception as e:
        log(f"ERROR: llama-server not reachable at {LLAMA_SERVER_URL}: {e}")
        log("Start it with: ./start_server.sh")
        sys.exit(1)

    # Local client for llama-server (no timeout — inference can be slow)
    llama = httpx.Client(timeout=httpx.Timeout(180))

    report_status(mgr, state, "running")

    tagged = 0
    for i, asset in enumerate(assets):
        asset_id = asset["id"]
        filename = asset["filename"]
        full_path = asset["full_path"]
        rel_path = asset.get("rel_path", filename)

        if str(asset_id) in processed:
            continue

        limit_msg = ""
        if args.max and tagged >= args.max:
            limit_msg = f" (--max={args.max})"

        log(f"[{tagged + 1}/{min(state['total'], args.max or state['total'])}{limit_msg}] #{asset_id} {filename}")

        result = tag_audio(full_path, filename, rel_path, llama)
        if result:
            if send_to_asset_manager(asset_id, result, mgr):
                tagged += 1
                state["tagged"] += 1
                log(f"  ✓ {result.get('tags', '')[:80]}")
        else:
            state["errors"] += 1

        processed.add(str(asset_id))
        state["processed"] = list(processed)
        state["runtime_s"] = int(time.time() - start_time)

        # Save state periodically
        if tagged % 10 == 0:
            save_state(state)
            report_status(mgr, state, "running")

        if args.max and tagged >= args.max:
            log(f"Reached --max={args.max}, stopping.")
            break

        time.sleep(DELAY_SEC)

    # Final save
    state["runtime_s"] = int(time.time() - start_time)
    save_state(state)
    report_status(mgr, state, "completed")

    log(f"Done! Tagged: {state['tagged']}, Errors: {state['errors']}, Runtime: {state['runtime_s']}s")


if __name__ == "__main__":
    main()
