#!/usr/bin/env python3
"""
Test Gemini Audio Tagging via native google-genai SDK.
Gemini 3.5 Flash supports audio natively — same multimodal interface as Gemma 4.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Load GOOGLE_API_KEY from Hermes .env
_HERMES_ENV = Path.home() / ".hermes" / ".env"
if _HERMES_ENV.exists():
    from dotenv import load_dotenv
    load_dotenv(_HERMES_ENV)

API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not API_KEY:
    print("❌ GOOGLE_API_KEY nicht gesetzt!")
    sys.exit(1)

from google import genai
from google.genai import types

MODEL_ID = "gemini-3.5-flash"

SYSTEM_PROMPT = """You are an audio analysis expert for game development. Listen carefully to the audio and respond ONLY with a JSON object:

{"tags": "tag1, tag2, tag3, tag4, tag5, tag6", "description": "Kurze deutsche Beschreibung des Sounds (1 Satz, game-dev-relevant)"}

Tags should be game-dev relevant categories (e.g. explosion, coin_loot, power_up, laser, ui_click, ambient_dark, magic_spell, footsteps, etc.).
Output ONLY the JSON, no other text."""

USER_PROMPT = "Analyze this audio clip for a game asset library. What kind of sound is this? Give 5-8 precise tags and a short German description."


def test_single_file(audio_path: Path) -> dict | None:
    """Test Gemini audio tagging on one file."""
    client = genai.Client(api_key=API_KEY)

    if not audio_path.exists():
        print(f"❌ Datei nicht gefunden: {audio_path}")
        return None

    print(f"📁 Audio: {audio_path.name} ({audio_path.stat().st_size / 1024:.1f} KB)")

    # --- Upload ---
    print("⏫ Upload zu Gemini Files API...")
    try:
        uploaded = client.files.upload(file=audio_path)
        print(f"   Uploaded: {uploaded.name} (state={uploaded.state})")
    except Exception as e:
        print(f"❌ Upload fehlgeschlagen: {e}")
        return None

    # Wait for processing if needed
    for _ in range(10):
        if uploaded.state.name != "PROCESSING":
            break
        time.sleep(1)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name == "FAILED":
        print(f"❌ File processing FAILED")
        _cleanup(client, uploaded.name)
        return None

    # --- Generate ---
    print(f"🤖 Sende an {MODEL_ID}...")
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[uploaded, USER_PROMPT],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        print(f"❌ API-Fehler: {e}")
        _cleanup(client, uploaded.name)
        return None

    # Cleanup uploaded file
    _cleanup(client, uploaded.name)

    text = response.text.strip()
    print(f"📝 Raw Response: {text[:300]}")

    try:
        result = json.loads(text)
        print(f"\n✅ ERFOLG!")
        print(f"   Tags:        {result.get('tags', '—')}")
        print(f"   Beschreibung: {result.get('description', '—')}")
        return result
    except json.JSONDecodeError:
        print(f"❌ Kein gültiges JSON. Versuche Regex-Extraktion...")
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                print(f"✅ JSON per Regex extrahiert!")
                print(f"   Tags:        {result.get('tags', '—')}")
                print(f"   Beschreibung: {result.get('description', '—')}")
                return result
            except json.JSONDecodeError:
                pass
        print(f"❌ JSON-Extraktion fehlgeschlagen.")
        return None


def _cleanup(client, file_name: str) -> None:
    try:
        client.files.delete(name=file_name)
    except Exception:
        pass


# --- Audio files to test ---
TEST_FILES = [
    "/mnt/orange/Music/100_Retro_Game_Sounds_SFX_[VOLUME1]_24 bits_44.1KHz_WAV_CC-BY-3.0_by_jalastram/Retro_Game_Sounds_SFX_01.wav",
    "/mnt/orange/Music/100_Retro_Game_Sounds_SFX_[VOLUME1]_24 bits_44.1KHz_WAV_CC-BY-3.0_by_jalastram/Retro_Game_Sounds_SFX_04.wav",
    "/mnt/orange/Music/100_Retro_Game_Sounds_SFX_[VOLUME1]_24 bits_44.1KHz_WAV_CC-BY-3.0_by_jalastram/Retro_Game_Sounds_SFX_08.wav",
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Einzelne Audio-Datei testen")
    parser.add_argument("--all", action="store_true", help="Alle Test-Dateien")
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    elif args.all:
        files = [Path(f) for f in TEST_FILES]
    else:
        files = [Path(TEST_FILES[0])]  # Default: erste Datei

    print(f"╔══════════════════════════════════════╗")
    print(f"║  Gemini Audio Tagger — Test Runner  ║")
    print(f"║  Model: {MODEL_ID:<28}║")
    print(f"╚══════════════════════════════════════╝")
    print()

    success = 0
    for f in files:
        result = test_single_file(f)
        if result:
            success += 1
        print()

    print(f"✅ {success}/{len(files)} erfolgreich")
