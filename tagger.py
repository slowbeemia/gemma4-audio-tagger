#!/usr/bin/env python3
"""
gemma4-audio-tagger — Batch audio tagging via Gemma 4 12B (LiteLLM + Ollama).

Uses google-genai SDK to connect to a local LiteLLM Proxy, which in turn
talks to Ollama running Gemma 4 on the AMD GPU.

Usage:
    uv run python tagger.py                  # tag next 10 untagged
    uv run python tagger.py --max 10         # tag 10 files
    uv run python tagger.py --max 1          # test one file
    uv run python tagger.py --dry-run        # show what would be tagged
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

from google import genai
from google.genai import types

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
MODEL_ID = os.getenv("MODEL_ID", "gemma4:12b")
ASSET_MANAGER_URL = os.getenv("ASSET_MANAGER_URL", "http://bob.fritz.box:8899")
DELAY_SEC = float(os.getenv("DELAY_SEC", "2"))
API_PAGE_SIZE = int(os.getenv("API_PAGE_SIZE", "50"))
PATH_MAP_FROM = os.getenv("PATH_MAP_FROM", "//orange/Assets")
PATH_MAP_TO = os.getenv("PATH_MAP_TO", "//orange/Assets")

STATE_FILE = Path(__file__).parent / ".tagger_state.json"
LOG_FILE = Path(__file__).parent / "tagger.log"

SYSTEM_PROMPT = """You are an audio analysis expert for game development. Listen carefully to the audio and respond ONLY with a JSON object:

{"tags": "tag1, tag2, tag3, tag4, tag5, tag6", "description": "Kurze deutsche Beschreibung des Sounds oder Musikstücks (1-2 Sätze, game-dev-relevant, inkl. akustischer Details wie Tonhöhenverlauf, Rhythmus/Pulsierung/Wiederholungen, menschliche Stimme/Vokale/Stottern oder Instrumentierung)"}

Tags should be game-dev relevant categories (e.g. explosion, ui_click, ambient_dark, background_music, voice_male, voice_female, vocal_grunt, stutter, vowel, synth, orchestral, etc.).
Do not hallucinate environmental sounds (like wind, rain, or forest) when hearing repetitive human vocalizations, stuttering, or raw voice waves. Classify them accurately as voice, speech, grunt, or vocal stutter.
Output ONLY the JSON, no other text."""


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


def convert_to_16khz_wav(filepath: Path) -> str | None:
    """Convert audio to 16kHz mono WAV (max 30 seconds), return temp file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(filepath),
            "-t", "30",
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


def tag_audio(asset_id: int, full_path: str, filename: str, rel_path: str) -> dict | None:
    """Send audio bytes to local Ollama directly, return parsed tags."""
    local_path = full_path.replace(PATH_MAP_FROM, PATH_MAP_TO, 1)
    local_path = local_path.replace("/", "\\")  # Normalize backslashes for Windows
    path = Path(local_path)
    if not path.exists():
        log(f"  SKIP: file not found: {local_path}")
        return None

    wav_path = convert_to_16khz_wav(path)
    if not wav_path:
        return None

    try:
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
    except Exception as e:
        log(f"  ERROR: Could not read WAV file: {e}")
        _cleanup_wav(wav_path)
        return None

    import base64
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Retry/Recovery logic for local model API
    max_retries = 3
    retry_delay = 5
    response_text = ""

    # Dynamic user prompt incorporating asset_id to prevent llama.cpp KV-cache reuse bug and handle music & vocal awareness (without leaking name to prevent cheating)
    dynamic_user_prompt = f"Analyze this audio clip (Request ID: {asset_id}) for a game asset library. What kind of sound or music is this? Identify if it is a sound effect (SFX), human vocalization or speech (such as a voice, grunt, stutter, vowel repetition, or shout), ambient background track, or music. Give 5-8 precise tags (including vocal characteristics if it is a voice/vocal) and a detailed German description describing its acoustic properties (like pitch modulation, rhythm/pulsation, repeating syllables/vowels, or instruments for music) and its game-dev use case. Do not try to guess based on the ID; you must listen to the audio waveform itself."

    for attempt in range(1, max_retries + 1):
        try:
            log(f"  Sende Audio-Asset an doitagain zur akustischen Analyse (Versuch {attempt}/{max_retries})...")
            
            resp = httpx.post(
                f"{OLLAMA_API_URL}/api/chat",
                json={
                    "model": MODEL_ID,
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": dynamic_user_prompt,
                            "images": [audio_b64]
                        }
                    ],
                    "options": {
                        "temperature": 0.0
                    },
                    "format": "json",
                    "think": False,
                    "stream": False
                },
                timeout=120.0
            )
            resp.raise_for_status()
            res_data = resp.json()
            response_text = res_data["message"]["content"]
            if response_text:
                break
        except Exception as e:
            log(f"  ERROR (Versuch {attempt}): API-Fehler: {e}")
            if attempt < max_retries:
                log(f"  Warte {retry_delay}s vor dem nächsten Versuch...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                log("  FATAL: Alle Versuche fehlgeschlagen.")
                _cleanup_wav(wav_path)
                return None

    _cleanup_wav(wav_path)

    if not response_text:
        log("  ERROR: Empty response from model")
        return None

    # Log raw output
    log(f"  [RAW] {repr(response_text[:400])}")

    # Find JSON in output (model may add surrounding text)
    match = re.search(r"\{[^{}]*\"tags\"[^{}]*\}", response_text, re.DOTALL)
    if not match:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            log(f"  ERROR: JSON parse failed: {e}")

    log(f"  ERROR: No valid JSON in output")
    return None


def send_to_asset_manager(asset_id: int, tags: dict, client_http: httpx.Client) -> bool:
    try:
        resp = client_http.post(
            f"{ASSET_MANAGER_URL}/api/audio-assets/{asset_id}/tag",
            json={
                "tags": tags.get("tags", ""),
                "description": tags.get("description", ""),
                "notes": "gemma4-12b-litellm",
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
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"tagged": 0, "errors": 0, "processed": [], "total": 0, "runtime_s": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Gemma 4 Audio Tagger (Ollama)")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    log("=" * 60)
    log("Gemma 4 Audio Tagger (Direct Ollama Backend)")
    log(f"Ollama URL: {OLLAMA_API_URL}")
    log(f"Model ID:   {MODEL_ID}")
    log(f"Asset Mgr:  {ASSET_MANAGER_URL}")

    # Quick connection check to Ollama
    log(f"Checking Ollama connection at {OLLAMA_API_URL}...")
    try:
        resp = httpx.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            log("Ollama connection: OK")
        else:
            log(f"WARNING: Ollama responded with status {resp.status_code}, continuing anyway...")
    except Exception as e:
        log(f"FATAL: Could not connect to Ollama: {e}")
        log("Please make sure Ollama is running first!")
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

        result = tag_audio(asset_id, full_path, filename, asset.get("rel_path", filename))
        if result:
            log(f"  Tags: {result.get('tags', '—')}")
            if result.get("description"):
                log(f"  Desc: {result['description']}")
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
