#!/bin/bash
# setup_llama_cpp.sh — Build llama.cpp with CUDA on WSL2/Ubuntu
# Run this ONCE on TheGrey WSL2. Requires: NVIDIA driver on Windows side.
set -euo pipefail

echo "=== Checking CUDA ==="
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi failed. Is the NVIDIA driver installed on Windows?"
    echo "WSL2 passes the Windows driver through — no Linux driver needed."
    echo "Check: https://developer.nvidia.com/cuda/wsl"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Installing build dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y pciutils build-essential cmake curl libcurl4-openssl-dev

echo ""
echo "=== Cloning llama.cpp ==="
LLAMA_DIR="$HOME/llama.cpp-gemma4"
if [ -d "$LLAMA_DIR" ]; then
    echo "Directory exists, pulling latest..."
    cd "$LLAMA_DIR"
    git pull
else
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    cd "$LLAMA_DIR"
fi

echo ""
echo "=== Building with CUDA ==="
cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=86

cmake --build build --config Release -j$(nproc) --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

echo ""
echo "=== Verifying build ==="
./build/bin/llama-server --version 2>&1 | head -3
./build/bin/llama-mtmd-cli --help 2>&1 | head -3

echo ""
echo "=== Done! ==="
echo "Binaries: $LLAMA_DIR/build/bin/"
echo "Next: run download_model.sh"
