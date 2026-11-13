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
    http_options={"base_url": LITELLM_PROXY_URL}
)

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

print("Sending audio request to LiteLLM (no headers)...")
try:
    response = client.models.generate_content(
        model="gemma4:12b",
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            "Analyze this audio clip. What kind of sound is this? Respond in one sentence."
        ],
    )
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
