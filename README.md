# Gemma 4 12B Audio Tagger

Batch audio tagging pipeline using Google's new **Gemma 4 12B** model — the first medium-sized model with **native audio understanding** (encoder-free architecture).

Runs locally on an **RTX 3060 12GB** via llama.cpp + CUDA. Tags game audio assets (SFX, music, ambient, voice) with practical German descriptions for the [Asset Manager](http://bob.fritz.box:8899).

## Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Audio file  │────▶│  llama-server   │────▶│  Asset       │
│  (WAV 16kHz) │     │  Gemma 4 12B    │     │  Manager API │
│              │     │  :8001          │     │  :8899       │
└──────────────┘     └─────────────────┘     └──────────────┘
       ▲                     │                      │
       │                     ▼                      ▼
   SMB Share           JSON tags +           Status dashboard
   //orange/Assets     description           on Bob
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 8GB VRAM | 12GB VRAM (RTX 3060+) |
| RAM | 16GB | 32GB |
| Storage | ~8GB (Q4_K_M) | ~12GB (Q6_K) |
| OS | WSL2 Ubuntu 24.04 | same |

**Tested on:** RTX 3060 12GB, Q4_K_M quantization (~7.8GB VRAM)

## Quick Start

### 1. Setup (TheGrey WSL2)

```bash
# One-time setup
git clone https://github.com/minionhermi/gemma4-audio-tagger.git
cd gemma4-audio-tagger

# Build llama.cpp with CUDA
./setup_llama_cpp.sh

# Download model + mmproj
./download_model.sh

# Install Python deps
uv sync
```

### 2. Mount SMB share

```bash
# WSL2: use drvfs (passes through Windows SMB client)
sudo mount -t drvfs '\\orange\Assets' /mnt/orange -o credentials=/etc/cifs-credentials,ro,uid=1000,gid=1000
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — set SMB_PASS and any custom paths
```

### 4. Start server

```bash
./start_server.sh
# API live at http://localhost:8001/v1/chat/completions
```

### 5. Run tagger

```bash
# Dry run — see what would be tagged
uv run python3 tagger.py --dry-run

# Tag 10 files (test run)
uv run python3 tagger.py --max 10

# Tag everything
uv run python3 tagger.py
```

## Model Details

Gemma 4 12B is Google DeepMind's newest open model with a **unified encoder-free architecture**:
- No separate vision or audio encoders
- Raw 16kHz audio → 40ms frames → linear projection → LLM backbone
- Max audio duration: **30 seconds**
- Apache 2.0 license

Two quantization options for 12GB VRAM:
| Quant | Size | Quality | Fit |
|-------|------|---------|-----|
| `UD-Q4_K_XL` | ~7.5 GB | Very good | ✅ Comfortable |
| `Q4_K_M` | ~7.8 GB | Good | ✅ fits |
| `Q5_K_M` | ~9.5 GB | Better | ⚠️ Tight |

Default: `UD-Q4_K_XL` (Unsloth Dynamic 4-bit, recommended)

## Files

| File | Purpose |
|------|---------|
| `setup_llama_cpp.sh` | Build llama.cpp with CUDA on WSL2 |
| `download_model.sh` | Download GGUF + mmproj from HuggingFace |
| `start_server.sh` | Launch llama-server with audio support |
| `tagger.py` | Main batch tagging script |
| `.env` | Configuration (copy from .env.example) |

## Audio Format

- Input: any format ffmpeg can read (.wav, .mp3, .ogg, .flac)
- Automatically converted to 16kHz mono WAV before sending
- Files > 30 seconds are skipped (model limit)
