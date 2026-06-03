#!/bin/bash
# start_server.sh — Launch llama-server with Gemma 4 12B + audio support
# Run this to start the OpenAI-compatible API on port 8001
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp-gemma4}"
MODELS_DIR="$HOME/gemma4-models"
QUANT="${1:-UD-Q4_K_XL}"

MODEL=$(ls "$MODELS_DIR"/gemma-4-12*-it-${QUANT}.gguf 2>/dev/null | head -1)
MMPROJ="$MODELS_DIR/mmproj-BF16.gguf"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Run download_model.sh first."
    exit 1
fi

if [ ! -f "$MMPROJ" ]; then
    echo "ERROR: mmproj not found: $MMPROJ"
    echo "Run download_model.sh first."
    exit 1
fi

echo "=== Starting llama-server ==="
echo "Model:  $MODEL"
echo "MMProj: $MMPROJ"
echo "API:    http://localhost:8001/v1/chat/completions"
echo ""

"$LLAMA_DIR/build/bin/llama-server" \
    --model "$MODEL" \
    --mmproj "$MMPROJ" \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    --ctx-size 32768 \
    --n-gpu-layers 99 \
    --alias "gemma-4-12b-it" \
    --port 8001 \
    --host 0.0.0.0 \
    --chat-template-kwargs '{"enable_thinking":false}'
