#!/usr/bin/env python3
"""
gemma4-audio-tagger — Batch audio tagging via Gemma 4 12B (llama.cpp).

Uses llama-mtmd-cli (the multimodal CLI that ships with llama.cpp) as a
subprocess. llama-server does NOT support input_audio — the CLI is the
supported path for Gemma 4 audio inference.

Usage:
    uv run python3 tagger.py                  # tag next 10 untagged
    uv run python3 tagger.py --max 10         # tag 10 files
    uv run python3 tagger.py --max 1          # test one file
    uv run python3 tagger.py --dry-run        # show what would be tagged
"""
from __future__ import annotations

import json
import os
import re
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

LLAMA_MTMD_CLI = os.getenv("LLAMA_MTMD_CLI", "llama-mtmd-cli")
LLAMA_MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    "/home/dirk/gemma4-models/gemma-4-12b-it-UD-Q4_K_XL.gguf",
)
LLAMA_MMPROJ_PATH = os.getenv(
    "LLAMA_MMPROJ_PATH",
    "/home/dirk/gemma4-models/mmproj-BF16.gguf",
)
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


def convert_to_16khz_wav(filepath: Path) -> str | None:
    """Convert audio to 16kHz mono WAV, return temp file path."""
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
        return tmp_path
    except Exception as e:
        log(f"  ERROR: ffmpeg conversion failed: {e}")
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return None


def tag_audio(full_path: str, filename: str, rel_path: str) -> dict | None:
    """Send audio to llama-mtmd-cli, return parsed tags."""
    local_path = full_path.replace(PATH_MAP_FROM, PATH_MAP_TO, 1)
    path = Path(local_path)
    if not path.exists():
        log(f"  SKIP: file not found: {local_path}")
        return None

    wav_path = convert_to_16khz_wav(path)
    if not wav_path:
        return None

    # Build prompt: system + user combined (llama-mtmd-cli takes one prompt string)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"

    cmd = [
        LLAMA_MTMD_CLI,
        "-m", LLAMA_MODEL_PATH,
        "--mmproj", LLAMA_MMPROJ_PATH,
        "--audio", wav_path,
        "--temp", "0.0",
        "-p", full_prompt,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        log(f"  ERROR: llama-mtmd-cli timed out (120s)")
        _cleanup_wav(wav_path)
        return None
    except FileNotFoundError:
        log(f"  ERROR: {LLAMA_MTMD_CLI} not found. Build llama.cpp first.")
        _cleanup_wav(wav_path)
        return None

    _cleanup_wav(wav_path)

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        log(f"  ERROR: llama-mtmd-cli exit={result.returncode}")
        if stderr:
            log(f"  [STDERR] {stderr[:300]}")
        return None

    # Log raw output
    log(f"  [RAW] {repr(stdout[:400])}")

    # Find JSON in output (model may add surrounding text)
    match = re.search(r"\{[^{}]*\"tags\"[^{}]*\}", stdout, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", stdout, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            log(f"  ERROR: JSON parse failed: {e}")

    log(f"  ERROR: No valid JSON in output")
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


def _cleanup_wav(path: str) -> None:
    try:
        os.unlink(path)
    except Exception:
        pass


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"tagged": 0, "errors": 0, "processed": [], "total": 0, "runtime_s": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 Audio Tagger (llama.cpp)")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    log("=" * 60)
    log("Gemma 4 Audio Tagger (llama-mtmd-cli)")
    log(f"CLI:     {LLAMA_MTMD_CLI}")
    log(f"Model:   {LLAMA_MODEL_PATH}")
    log(f"MMProj:  {LLAMA_MMPROJ_PATH}")
    log(f"Asset Mgr: {ASSET_MANAGER_URL}")

    # Quick check: is llama-mtmd-cli available?
    try:
        r = subprocess.run([LLAMA_MTMD_CLI, "--version"], capture_output=True, timeout=5)
        if r.returncode == 0:
            log(f"llama-mtmd-cli: OK")
        else:
            log(f"WARNING: {LLAMA_MTMD_CLI} --version failed, continuing anyway...")
    except FileNotFoundError:
        log(f"FATAL: {LLAMA_MTMD_CLI} not found in PATH!")
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
    tried = 0  # count attempts, stop after max_count tries
    for asset in assets:
        if tried >= max_count:
            break

        asset_id = asset["id"]
        if str(asset_id) in processed:
            continue

        tried += 1
        filename = asset["filename"]
        full_path = asset["full_path"]

        log(f"[{tagged}/{tried}/{max_count}] #{asset_id} {filename}")

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

        if tagged % 5 == 0 and tagged > 0:
            save_state(state)

        time.sleep(DELAY_SEC)

    state["runtime_s"] = int(time.time() - start_time)
    save_state(state)
    log(f"Done! Tagged: {tagged}, Errors: {state['errors']}, Runtime: {state['runtime_s']}s")


if __name__ == "__main__":
    main()
