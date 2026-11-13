# Gemma 4 Audio Tagger (PowerShell + Windows)

Batch audio tagging via **Gemma 4 12B** running locally on Windows (Ollama + LiteLLM Proxy + AMD GPU ROCm).

## Architektur

```
[ Remote-Node: bob ] (Führt Audio-Tagger aus via Google GenAI SDK)
         │
         ▼ (Netzwerk-Request via HTTP auf Port 9379)
[ Host: doitagain ]
   ├── ① LiteLLM Proxy (Port 9379) ──► Übersetzt Google GenAI Format in Ollama API
   ├── ② Ollama Backend (Port 11434) ──► Verwaltet die AMD-Modellausführung via ROCm
   └── ③ Gemma 4 12B (Lokales Modell) ──► Berechnet die native Multimodal-Audio-Analyse
```

Audio wird per ffmpeg → 16kHz mono WAV → base64 → `Part.from_bytes` an den lokalen LiteLLM Proxy gesendet, welcher die Anfrage übersetzt und an Ollama weiterleitet.

## Quick Start (Windows / doitagain)

1. **Einrichten (einmalig):**
   Öffne eine PowerShell in diesem Ordner und führe das Setup-Skript aus:
   ```powershell
   .\Setup-Tagger.ps1
   ```
   Dieses Skript:
   - Synchronisiert die Python-Abhängigkeiten via `uv sync`.
   - Lädt das `gemma4:12b` Modell via Ollama herunter.
   - Prüft, ob `ffmpeg` installiert ist.

2. **Dienste starten:**
   Starte die Backend-Dienste (Ollama und den LiteLLM Proxy):
   ```powershell
   .\Start-Services.ps1
   ```
   Dies startet Ollama im Hintergrund (falls nicht aktiv) und öffnet ein neues PowerShell-Fenster für den LiteLLM-Proxy auf Port `9379`.

3. **Konfiguration anpassen:**
   Erstelle deine `.env`-Datei:
   ```powershell
   Copy-Item .env.example .env
   ```
   Passe ggf. den `LITELLM_PROXY_URL` (z. B. auf `http://192.168.178.82:9379` falls du von einem Remote-Node wie `bob` zugreifst) und den `ASSET_MANAGER_URL` an.

4. **Test & Batch:**
   Führe den Tagger aus:
   ```powershell
   # Dry-Run (zeigt an, welche Dateien getaggt würden)
   uv run python tagger.py --dry-run

   # Test mit einem Asset
   uv run python tagger.py --max 1 --reset

   # Batch-Verarbeitung von 100 Assets
   uv run python tagger.py --max 100
   ```

## Voraussetzungen auf Windows

| Komponente | Empfohlener Installationsweg |
|------------|------------------------------|
| **Ollama** | [ollama.com](https://ollama.com/) (Unterstützt AMD ROCm automatisch) |
| **uv** | `winget install astral-sh.uv` |
| **ffmpeg** | `winget install Gyan.FFmpeg` |
