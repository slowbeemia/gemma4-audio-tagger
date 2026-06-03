#!/usr/bin/env python3
"""
gemma4-audio-tagger — Batch audio tagging via Gemma 4 12B on llama.cpp.

Sends audio files to a local llama-server (OpenAI-compatible API),
gets tags + descriptions back, and reports to the Asset Manager API.

Usage:
    uv run python3 tagger.py                  # tag all untagged audio
    uv run python3 tagger.py --max 10         # tag 10 files, then stop
    uv run python3 tagger.py --force          # retag already-tagged files
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
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
DELAY_SEC = float(os.getenv("DELAY_SEC", "2"))
AUDIO_EXTENSIONS = set(
    ext.strip().lower()
    for ext in os.getenv("AUDIO_EXTENSIONS", ".wav,.mp3,.ogg,.flac").split(",")
    if ext.strip()
)

SMB_SHARE = f"//{os.getenv('SMB_HOST', 'orange')}/Assets"
SMB_USER = os.getenv("SMB_USER", "")
SMB_PASS = os.getenv("SMB_PASS", "")

STATE_FILE = Path(__file__).parent / ".tagger_state.json"
LOG_FILE = Path(__file__).parent / "tagger.log"

# --- Audio prompt (30s max for Gemma 4 12B) --------------------------------

SYSTEM_PROMPT = """You are an audio analysis expert for game development. Your task is to tag and describe short sound effect / music clips.

For each audio clip, output a JSON object with these fields:
- "tags": array of 3-8 lowercase tags (e.g. ["explosion", "impact", "boom", "deep", "cinematic"])
- "description": short German description (1 sentence, practical, game-dev focused)
- "category": one of [sfx, music, ambient, voice, ui, foley]
- "duration_seconds": estimated duration as a number
- "instruments": array of instruments if music, otherwise empty array

Output ONLY the JSON object, no other text."""

USER_PROMPT_TEMPLATE = """Tag this audio clip for a game asset library.

Audio is part of: {relative_path}

Output the JSON object now."""


def log(msg: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"tagged": 0, "errors": 0, "processed": set(), "total": 0, "runtime_s": 0}


def save_state(state: dict) -> None:
    # Convert set to list for JSON
    state_copy = {**state, "processed": list(state["processed"])}
    STATE_FILE.write_text(json.dumps(state_copy, indent=2))


def report_status(state: dict, status: str = "running") -> None:
    """Report progress to Asset Manager API on Bob."""
    try:
        resp = httpx.post(
            f"{ASSET_MANAGER_URL}/api/tagging-status",
            json={
                "host": "thegrey-gemma4",
                "model": "gemma-4-12b-it",
                "status": status,
                "current": state["tagged"],
                "total": state["total"],
                "errors": state["errors"],
                "runtime_s": state["runtime_s"],
            },
            timeout=5,
        )
        resp.raise_for_status()
    except Exception as e:
        log(f"Status report failed: {e}")


def convert_to_wav_16khz(audio_path: Path) -> bytes:
    """Convert any audio file to 16kHz mono WAV bytes using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio_path),
                "-ar", "16000", "-ac", "1",
                "-sample_fmt", "s16",
                "-loglevel", "error",
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        wav_data = tmp_path.read_bytes()
        return wav_data
    finally:
        tmp_path.unlink(missing_ok=True)


def tag_audio(audio_path: str, relative_path: str, client: httpx.Client) -> dict | None:
    """Send audio to llama-server, get tags back."""
    full_path = Path(audio_path)
    if not full_path.exists():
        log(f"  SKIP: file not found: {audio_path}")
        return None

    # Convert to 16kHz WAV
    try:
        wav_bytes = convert_to_wav_16khz(full_path)
    except subprocess.CalledProcessError as e:
        log(f"  ERROR: ffmpeg conversion failed: {e.stderr.decode()[:200]}")
        return None

    # Check 30s limit
    wav_size = len(wav_bytes)
    # WAV header = 44 bytes, 16-bit mono 16kHz = 32000 bytes/sec
    audio_duration = (wav_size - 44) / 32000 if wav_size > 44 else 0
    if audio_duration > 30:
        log(f"  SKIP: audio too long ({audio_duration:.1f}s > 30s max)")
        return None

    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    # Send to llama-server
    payload = {
        "model": "gemma-4-12b-it",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT_TEMPLATE.format(relative_path=relative_path)},
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
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        log(f"  ERROR: llama-server request failed: {e}")
        return None

    # Parse response
    try:
        content = data["choices"][0]["message"]["content"]
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        result = json.loads(content)
        return result
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        log(f"  ERROR: failed to parse response: {e}")
        log(f"  Raw: {content[:300]}")
        return None


def send_to_asset_manager(relative_path: str, tags: dict) -> bool:
    """Send tags to Asset Manager API."""
    try:
        resp = httpx.post(
            f"{ASSET_MANAGER_URL}/api/assets/tag",
            json={
                "path": relative_path,
                "tags": tags.get("tags", []),
                "description": tags.get("description", ""),
                "category": tags.get("category", "sfx"),
                "duration_seconds": tags.get("duration_seconds", 0),
                "instruments": tags.get("instruments", []),
                "tagger": "gemma4-12b",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        log(f"  ERROR: Asset Manager API failed: {e}")
        return False


def collect_audio_files(smb_mount: str) -> list[tuple[str, str]]:
    """Walk the SMB mount and find audio files. Returns (absolute_path, relative_path)."""
    base = Path(smb_mount)
    if not base.exists():
        log(f"ERROR: SMB mount not found: {smb_mount}")
        log("Mount with: sudo mount -t cifs //orange/Assets /mnt/orange -o credentials=/etc/cifs-credentials")
        return []

    files = []
    for f in base.rglob("*"):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS:
            abs_path = str(f)
            rel_path = str(f.relative_to(base))
            files.append((abs_path, rel_path))

    return sorted(files)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 12B Audio Tagger")
    parser.add_argument("--max", type=int, default=0, help="Max files to tag (0 = all)")
    parser.add_argument("--force", action="store_true", help="Retag already-tagged files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be tagged")
    parser.add_argument("--path", type=str, default="/mnt/orange", help="Audio directory")
    args = parser.parse_args()

    log("=" * 60)
    log("Gemma 4 12B Audio Tagger starting")
    log(f"Server: {LLAMA_SERVER_URL}")
    log(f"Asset Manager: {ASSET_MANAGER_URL}")

    # Load state
    state = load_state()
    if args.max:
        state["tagged"] = 0
        state["errors"] = 0
        state["processed"] = set()
    else:
        state["processed"] = set(state.get("processed", []))
    start_time = time.time()

    # Collect files
    audio_files = collect_audio_files(args.path)
    state["total"] = len(audio_files)
    log(f"Found {len(audio_files)} audio files in {args.path}")

    if args.dry_run:
        for abs_path, rel_path in audio_files[:20]:
            print(f"  {rel_path}")
        if len(audio_files) > 20:
            print(f"  ... and {len(audio_files) - 20} more")
        return

    # Health check
    client = httpx.Client(timeout=httpx.Timeout(10))
    try:
        resp = client.get(f"{LLAMA_SERVER_URL}/health")
        log(f"llama-server health: {resp.json()}")
    except Exception as e:
        log(f"ERROR: llama-server not reachable: {e}")
        log("Start it with: ./start_server.sh")
        sys.exit(1)

    report_status(state, "running")

    # Tag files
    tagged = 0
    for abs_path, rel_path in audio_files:
        if rel_path in state["processed"] and not args.force:
            continue

        log(f"[{tagged + 1}/{min(state['total'], args.max or state['total'])}] {rel_path}")

        result = tag_audio(abs_path, rel_path, client)
        if result:
            if send_to_asset_manager(rel_path, result):
                tagged += 1
                state["tagged"] += 1
        else:
            state["errors"] += 1

        state["processed"].add(rel_path)
        state["runtime_s"] = int(time.time() - start_time)

        # Save state every BATCH_SIZE files
        if tagged % BATCH_SIZE == 0:
            save_state(state)
            report_status(state, "running")

        if args.max and tagged >= args.max:
            log(f"Reached --max={args.max}, stopping.")
            break

        time.sleep(DELAY_SEC)

    # Final save
    state["runtime_s"] = int(time.time() - start_time)
    save_state(state)
    report_status(state, "completed")

    log(f"Done! Tagged: {state['tagged']}, Errors: {state['errors']}, "
         f"Runtime: {state['runtime_s']}s")


if __name__ == "__main__":
    main()
