#!/bin/bash
# download_model.sh — Download Gemma 4 12B GGUF + mmproj from Unsloth
# Run this AFTER setup_llama_cpp.sh
set -euo pipefail

MODELS_DIR="$HOME/gemma4-models"
mkdir -p "$MODELS_DIR"

# --- Choose quantization ---
# Q4_K_M:  ~7.8 GB  — fits RTX 3060 12GB with KV cache room
# Q5_K_M:  ~9.5 GB  — slightly better quality, tight on 12GB
# Q6_K:    ~11.5 GB — best quality for 12GB, very tight
# UD-Q4_K_XL: ~7.5 GB — Unsloth Dynamic 4-bit (recommended by Unsloth)
QUANT="${1:-UD-Q4_K_XL}"

echo "=== Downloading Gemma 4 12B ($QUANT) ==="
echo "Target: $MODELS_DIR"

# Use uvx to run hf CLI without installing anything system-wide
if command -v uvx &>/dev/null; then
    HF="uvx --from huggingface_hub hf"
elif command -v uv &>/dev/null; then
    HF="uvx --from huggingface_hub hf"
elif command -v hf &>/dev/null; then
    HF="hf"
else
    echo "ERROR: No way to run 'hf' CLI. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Download model + mmproj from Unsloth
$HF download "unsloth/gemma-4-12B-it-GGUF" \
    --include "*mmproj-BF16*" \
    --include "*${QUANT}*" \
    --local-dir "$MODELS_DIR"

echo ""
echo "=== Downloaded files ==="
ls -lh "$MODELS_DIR"

echo ""
echo "=== Done! ==="
echo "Model: $MODELS_DIR/gemma-4-12B-it-${QUANT}.gguf"
echo "MMProj: $MODELS_DIR/mmproj-BF16.gguf"
echo "Next: run start_server.sh"
