# Gemma 4 Audio Tagger

Batch audio tagging via **Gemma 4 12B** auf TheGrey (llama.cpp + RTX 3060).

## Architektur

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Audio file  │────▶│  llama-server   │────▶│  Asset       │
│  (beliebiges  │     │  Gemma 4 12B    │     │  Manager API │
│   Format)     │     │  :8001          │     │  :8899       │
└──────────────┘     └─────────────────┘     └──────────────┘
       ▲                     │                      │
       │                     ▼                      ▼
   /mnt/orange          JSON tags +           Status auf Bob
   (Orange SMB)         description
```

Audio wird per ffmpeg → 16kHz mono WAV → base64 → `input_audio` Content Block an llama-server gesendet.

## Quick Start (TheGrey)

```bash
cd ~/Projekte/gemma4-audio-tagger
git pull

# 1. Dependencies (nur httpx + python-dotenv, kein torch!)
uv sync

# 2. .env erstellen
cp .env.example .env
# Prüfe: LLAMA_SERVER_URL, ASSET_MANAGER_URL, PATH_MAP_TO

# 3. llama-server starten (vorher!)
./start_server.sh

# 4. Test mit einem Asset
uv run python3 tagger.py --max 1 --reset

# 5. Batch
uv run python3 tagger.py --max 100
```

## Voraussetzungen auf TheGrey

| Was | Befehl |
|-----|--------|
| llama.cpp mit CUDA | `./setup_llama_cpp.sh` |
| GGUF Model + mmproj | `./download_model.sh` → `/home/dirk/gemma4-models/` |
| Orange SMB Mount | `sudo mount -t drvfs '\\orange\Assets' /mnt/orange` |
| llama-server läuft | `./start_server.sh` → `:8001` |

## Model

- **Datei:** `/home/dirk/gemma4-models/gemma-4-12b-it-UD-Q4_K_XL.gguf`
- **MMProj:** `/home/dirk/gemma4-models/mmproj-BF16.gguf`
- **Quant:** UD-Q4_K_XL (Unsloth Dynamic, ~7.5 GB)
- **VRAM:** ~8 GB auf RTX 3060 12GB

## Audio-Format

- Input: beliebig (WAV, MP3, OGG, FLAC)
- Automatisch via ffmpeg → 16kHz mono WAV → base64
- Files > 30s werden vom Modell nicht verarbeitet (Hardware-Limit)

## Alternativen

- `gemini_tagger.py` — Gemini 3.5 Flash Cloud API (für Bob, teurer)
- `test_gemini_audio.py` — Einzeltest Gemini Audio
