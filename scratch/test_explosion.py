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

# Use the exact UNC path
path = Path(r"\\orange\Assets\Music\Humble_Music_SFX2_Bundle\Explosion Sound FX - Explosion Volume I\TomWinandySFX_Explosions Volume I_MediumExplosion_10.wav")
print("Checking if path exists:", path.exists())

wav_path = convert_to_16khz_wav(path)
if not wav_path:
    print("Failed to convert WAV")
    sys.exit(1)

with open(wav_path, "rb") as f:
    audio_bytes = f.read()

os.unlink(wav_path)

audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

SYSTEM_PROMPT = """You are an audio analysis expert for game development. Listen carefully to the audio and respond ONLY with a JSON object:

{"tags": "tag1, tag2, tag3, tag4, tag5, tag6", "description": "Kurze deutsche Beschreibung des Sounds (1-2 Sätze, game-dev-relevant, inkl. akustischer Details wie Tonhöhenverlauf (aufsteigend/absteigend) und Rhythmus/Pulsierung/Wiederholungen/Tremolo)"}

Tags should be game-dev relevant categories (e.g. explosion, coin_loot, power_up, laser, ui_click, ambient_dark, magic_spell, footsteps, etc.).
Output ONLY the JSON, no other text."""

# Unique prompt including file name or random seed to bypass KV cache
USER_PROMPT = "Analyze this audio clip (File: TomWinandySFX_Explosions Volume I_MediumExplosion_10.wav) for a game asset library. What kind of sound is this? Give 5-8 precise tags and a detailed German description describing its acoustic properties (like pitch modulation, rising/falling tone, and whether it has a single pulse or multiple pulses/pulsation/rhythm/repetitions) and its game-dev use case."

print("Sending direct audio request to Ollama...")
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
    print("Ollama response status:", resp.status_code)
    print("Ollama response:", resp.json().get("message", {}).get("content"))
except Exception as e:
    print("Error:", e)
