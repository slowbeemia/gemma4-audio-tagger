import base64
import httpx
from pathlib import Path
import subprocess
import tempfile
import sys
import os

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

Output ONLY the JSON, no other text."""

USER_PROMPT = "Analyze this audio clip for a game asset library. What kind of sound is this? Give 5-8 precise tags and a short German description."

print("Sending direct audio request to Ollama with think: False...")
try:
    resp = httpx.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": "gemma4:12b",
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": USER_PROMPT,
                    "images": [audio_b64]
                }
            ],
            "options": {
                "temperature": 0.0
            },
            "think": False,
            "stream": False
        },
        timeout=120.0
    )
    print("Ollama direct response status:", resp.status_code)
    data = resp.json()
    print("Ollama response content:", data.get("message", {}).get("content"))
    print("Full Ollama response keys:", data.keys())
    if "message" in data:
        print("Message fields:")
        for k, v in data["message"].items():
            if k != "content":
                print(f"  {k}: {repr(v)[:200]}")
except Exception as e:
    print("Error:", e)
