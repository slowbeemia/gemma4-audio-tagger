from google import genai
from google.genai import types
from pathlib import Path
import subprocess
import tempfile
import sys
import os

LITELLM_PROXY_URL = "http://localhost:9379"
client = genai.Client(
    api_key="local-proxy-key",
    http_options=types.HttpOptions(
        base_url=LITELLM_PROXY_URL,
        headers={"x-litellm-disable-callbacks": "true"},
        httpx_client=httpx_client if 'httpx_client' in globals() else None
    )
)

# We can also just use httpx directly to avoid any client wrapper issues
import httpx

def convert_to_16khz_wav(filepath: Path) -> str | None:
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
        print("ffmpeg conversion failed:", e)
        return None

path = Path("//orange/Assets/Music/100_Retro_Game_Sounds_SFX_[VOLUME1]_24 bits_44.1KHz_WAV_CC-BY-3.0_by_jalastram/Retro_Game_Sounds_SFX_21.wav")
wav_path = convert_to_16khz_wav(path)
if not wav_path:
    sys.exit(1)

with open(wav_path, "rb") as f:
    audio_bytes = f.read()

os.unlink(wav_path)

import base64
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

SYSTEM_PROMPT = """You are an audio analysis expert for game development. Listen carefully to the audio and respond ONLY with a JSON object:

{"tags": "tag1, tag2, tag3, tag4, tag5, tag6", "description": "Kurze deutsche Beschreibung des Sounds (1 Satz, game-dev-relevant)"}

Tags should be game-dev relevant categories (e.g. explosion, coin_loot, power_up, laser, ui_click, ambient_dark, magic_spell, footsteps, etc.).
Output ONLY the JSON, no other text."""

USER_PROMPT = "Analyze this audio clip for a game asset library. What kind of sound is this? Give 5-8 precise tags and a short German description."

print("Sending audio request via streaming to LiteLLM...")
try:
    # We call the OpenAI chat completions endpoint directly with streaming
    with httpx.stream(
        "POST",
        "http://localhost:9379/v1/chat/completions",
        json={
            "model": "gemma4:12b",
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": USER_PROMPT
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 150, # Set a safe limit on tokens to prevent infinite loops
            "stream": True
        },
        headers={"x-litellm-disable-callbacks": "true"},
        timeout=120.0
    ) as response:
        print(f"Status: {response.status_code}")
        for chunk in response.iter_lines():
            if chunk:
                # print raw chunk
                print(chunk)
except Exception as e:
    print("Error:", e)
