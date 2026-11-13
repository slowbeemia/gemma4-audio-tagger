import base64
import httpx
from pathlib import Path
import subprocess
import tempfile
import sys
import os

LITELLM_PROXY_URL = "http://localhost:9379"

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

audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

SYSTEM_PROMPT = """You are an audio analysis expert for game development. Listen carefully to the audio and respond ONLY with a JSON object:

{"tags": "tag1, tag2, tag3, tag4, tag5, tag6", "description": "Kurze deutsche Beschreibung des Sounds (1 Satz, game-dev-relevant)"}

Tags should be game-dev relevant categories (e.g. explosion, coin_loot, power_up, laser, ui_click, ambient_dark, magic_spell, footsteps, etc.).
Output ONLY the JSON, no other text."""

USER_PROMPT = "Analyze this audio clip for a game asset library. What kind of sound is this? Give 5-8 precise tags and a short German description."

print("Sending audio request as image_url to LiteLLM completions...")
try:
    resp = httpx.post(
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
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{audio_b64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 500,
            "response_format": {"type": "json_object"},
            "stream": False
        },
        headers={"x-litellm-disable-callbacks": "true"},
        timeout=120.0
    )
    print("Status:", resp.status_code)
    print("Response JSON:", resp.json())
except Exception as e:
    print("Error:", e)
