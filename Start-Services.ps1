# Start-Services.ps1 — Start Ollama and LiteLLM Proxy on Windows
# Exposes the local Gemma 4 model via an OpenAI-compatible proxy on port 9379.

Write-Host "=== Starting local services for Gemma 4 Audio Tagger ===" -ForegroundColor Cyan

# 1. Check if Ollama is running on port 11434
$OLLAMA_PORT = 11434
$ollamaRunning = $false
$tcp = New-Object System.Net.Sockets.TcpClient
try {
    $tcp.Connect("127.0.0.1", $OLLAMA_PORT)
    $ollamaRunning = $true
    $tcp.Close()
} catch {
    # Port not open
}

if ($ollamaRunning) {
    Write-Host "[OK] Ollama is already running on port $OLLAMA_PORT." -ForegroundColor Green
} else {
    Write-Host "[INFO] Ollama is not running. Launching Ollama serve..." -ForegroundColor Yellow
    if (Get-Command "ollama" -ErrorAction SilentlyContinue) {
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Minimized
        Write-Host "Waiting 5 seconds for Ollama to initialize..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    } else {
        Write-Host "[ERROR] Ollama executable not found! Please run Ollama manually." -ForegroundColor Red
        Exit
    }
}

Write-Host "`n=== Services started! ===" -ForegroundColor Green
Write-Host "Ollama is listening at http://localhost:$OLLAMA_PORT" -ForegroundColor Green
Write-Host "You can now run: uv run python tagger.py" -ForegroundColor Green
