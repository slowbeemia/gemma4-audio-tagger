# Setup-Tagger.ps1 — Setup dependencies and local models for Gemma 4 Audio Tagger
# Run this once to initialize the environment under Windows.

Write-Host "=== Setting up Gemma 4 Audio Tagger ===" -ForegroundColor Cyan

# 1. Check if Ollama is installed
if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
    Write-Host "[OK] Ollama is installed." -ForegroundColor Green
    Write-Host "Pulling Gemma 4 12B model (this may take a while depending on internet speed)..." -ForegroundColor Yellow
    ollama pull gemma4:12b
} else {
    Write-Host "[ERROR] Ollama CLI not found in PATH!" -ForegroundColor Red
    Write-Host "Please download and install Ollama for Windows from: https://ollama.com/" -ForegroundColor Yellow
}

# 2. Check if uv is installed
if (Get-Command "uv" -ErrorAction SilentlyContinue) {
    Write-Host "[OK] uv is installed. Syncing Python dependencies..." -ForegroundColor Green
    uv sync
} else {
    Write-Host "[ERROR] uv not found in PATH!" -ForegroundColor Red
    Write-Host "Please install uv (e.g. via winget: 'winget install astral-sh.uv' or PowerShell: 'irm https://astral.sh/uv/install.ps1 | iex')" -ForegroundColor Yellow
}

# 3. Check if ffmpeg is installed
if (Get-Command "ffmpeg" -ErrorAction SilentlyContinue) {
    Write-Host "[OK] ffmpeg is installed (needed for 16kHz WAV conversion)." -ForegroundColor Green
} else {
    Write-Host "[WARN] WARNING: ffmpeg was not found in your PATH!" -ForegroundColor Yellow
    Write-Host "The tagger needs ffmpeg to convert audio files. Please install it (e.g., 'winget install Gyan.FFmpeg')." -ForegroundColor Yellow
}

Write-Host "`n=== Setup Complete! ===" -ForegroundColor Green
Write-Host "Next, run .\Start-Services.ps1 to launch Ollama and LiteLLM Proxy." -ForegroundColor Green
