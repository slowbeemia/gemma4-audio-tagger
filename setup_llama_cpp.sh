#!/bin/bash
# setup_llama_cpp.sh — Build llama.cpp with CUDA on WSL2/Ubuntu
# Run this ONCE on TheGrey WSL2. Requires: NVIDIA driver on Windows side.
set -euo pipefail
set -x

echo "=== Checking CUDA ==="
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi failed. Is the NVIDIA driver installed on Windows?"
    echo "WSL2 passes the Windows driver through — no Linux driver needed."
    echo "Check: https://developer.nvidia.com/cuda/wsl"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$GPU_INFO"

# Warn if VRAM < 8GB — 12B needs at least Q4_K_M (~7.8GB)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -lt 8000 ]; then
    echo ""
    echo "⚠️  WARNING: GPU has ${VRAM_MB}MB VRAM — too small for Gemma 4 12B (needs 8GB+)."
    echo "   Consider using Gemma 4 E4B instead (~5.5GB at 4-bit)."
    echo "   Press Ctrl+C to abort, or Enter to continue anyway."
    read -r
fi

echo ""
echo "=== Installing CUDA Toolkit (WSL2) ==="
if ! command -v nvcc &>/dev/null; then
    # Ubuntu 24.04 WSL2: install via apt
    sudo apt-get install -y -qq nvidia-cuda-toolkit 2>/dev/null || {
        echo "Trying NVIDIA official repo..."
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
        sudo apt-get update -qq
        sudo apt-get install -y cuda-toolkit-12-8
        rm -f cuda-keyring_1.1-1_all.deb
    }
fi
echo "CUDA: $(nvcc --version 2>&1 | grep release)"

# Detect GPU arch for cmake
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "GPU Compute Capability: $CUDA_ARCH"

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
echo "=== Building with CUDA (arch=$CUDA_ARCH) ==="
# Ensure nvcc is findable — CUDA toolkit sometimes installs to non-standard locations
if [ -f /usr/local/cuda/bin/nvcc ]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
    export PATH="/usr/local/cuda/bin:$PATH"
elif [ -f /usr/bin/nvcc ]; then
    export CUDACXX=/usr/bin/nvcc
fi
echo "nvcc: $(command -v nvcc)"

cmake -B build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON \
    "-DCMAKE_CUDA_COMPILER=$CUDACXX" \
    "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"

cmake --build build --config Release -j"$(nproc)" --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

echo ""
echo "=== Verifying build ==="
./build/bin/llama-server --version 2>&1 | head -3
./build/bin/llama-mtmd-cli --help 2>&1 | head -3

echo ""
echo "=== Done! ==="
echo "Binaries: $LLAMA_DIR/build/bin/"
echo "Next: run download_model.sh"
