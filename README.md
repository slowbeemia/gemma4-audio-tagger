# Gemma 4 Audio Tagger

Batch audio tagging pipeline — jetzt mit **Gemini 3.5 Flash** (Google Cloud) auf Bob, kein GPU-Zwang mehr.

Der originale `tagger.py` (HF Transformers, Gemma 4 12B lokal) existiert weiter für TheGrey/RTX 3060. Auf Bob läuft `gemini_tagger.py` via Google API — gleiche Audio-Schnittstelle, gleiches Prompt-Format.

## Architektur

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Audio file  │────▶│  Gemini 3.5     │────▶│  Asset       │
│  (WAV, MP3)  │     │  Flash (Google)  │     │  Manager API │
│              │     │  Files API       │     │  :8899       │
└──────────────┘     └─────────────────┘     └──────────────┘
       ▲                     │                      │
       │                     ▼                      ▼
   /mnt/orange          JSON tags +           Status dashboard
   (Orange SMB)         description           auf Bob
```

## Quick Start (Bob — Gemini Cloud)

```bash
cd ~/Projekte/gemma4-audio-tagger

# 1. Dependencies
uv sync

# 2. .env (GOOGLE_API_KEY wird aus ~/.hermes/.env geladen)
cp .env.example .env

# 3. Dry-run
uv run python3 gemini_tagger.py --dry-run

# 4. Tag 10 files
uv run python3 gemini_tagger.py --max 10

# 5. Status
uv run python3 gemini_tagger.py --status
```

## Quick Start (TheGrey — Lokales Gemma 4 12B)

Der originale `tagger.py` nutzt HuggingFace Transformers mit 8-bit Quantisierung.
Läuft nur auf TheGrey (RTX 3060 12GB, WSL2).

```bash
# Auf TheGrey:
cd gemma4-audio-tagger
uv sync
cp .env.example .env  # ASSET_MANAGER_URL anpassen falls nötig
uv run python3 tagger.py --dry-run
uv run python3 tagger.py --max 1
```

## Files

| Datei | Zweck |
|-------|-------|
| `gemini_tagger.py` | Batch-Tagger via Gemini 3.5 Flash Cloud API (Bob) |
| `tagger.py` | Batch-Tagger via lokales Gemma 4 12B (TheGrey, GPU) |
| `test_gemini_audio.py` | Einzeltest: Gemini Audio-Tagging ohne Asset Manager |
| `setup_llama_cpp.sh` | llama.cpp Build-Script (Legacy, für tagger.py nicht nötig) |
| `download_model.sh` | GGUF Download (Legacy) |
| `start_server.sh` | llama-server Start (Legacy) |
| `.env` | Konfiguration (aus .env.example kopieren) |

## Model Details

### Gemini 3.5 Flash (Cloud)
- Läuft auf Bob, keine GPU nötig
- Native Audio-Unterstützung via Files API
- Gleiche multimodale Schnittstelle wie Gemma 4
- Free Tier: 1.500 Requests/Tag

### Gemma 4 12B (Lokal)
- Läuft auf TheGrey RTX 3060 12GB
- 8-bit Quantisierung (~12 GB VRAM)
- Google DeepMind, Apache 2.0
- Encoder-freie Architektur: 16kHz Audio → 40ms Frames → LLM
