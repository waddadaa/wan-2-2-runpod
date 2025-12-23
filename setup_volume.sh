#!/bin/bash
# ==============================================================================
# WAN 2.2 Network Volume Setup Script
# Run this on a RunPod GPU pod with the network volume mounted at /workspace
# ==============================================================================

set -e

echo "=========================================="
echo "WAN 2.2 Network Volume Setup"
echo "=========================================="

# Configuration
WORKSPACE="/workspace"
MODELS_DIR="${WORKSPACE}/models"
ENV_DIR="${WORKSPACE}/env"
WAN_REPO="https://github.com/Wan-Video/Wan2.2.git"

# Hugging Face model IDs
declare -A MODELS=(
    ["Wan2.2-T2V-A14B"]="Wan-AI/Wan2.2-T2V-A14B"
    ["Wan2.2-I2V-A14B"]="Wan-AI/Wan2.2-I2V-A14B"
    ["Wan2.2-TI2V-5B"]="Wan-AI/Wan2.2-TI2V-5B"
    ["Wan2.2-S2V-14B"]="Wan-AI/Wan2.2-S2V-14B"
    ["Wan2.2-Animate-14B"]="Wan-AI/Wan2.2-Animate-14B"
)

# ==============================================================================
# Create directories
# ==============================================================================
echo "[1/5] Creating directories..."
mkdir -p "${MODELS_DIR}"
mkdir -p "${ENV_DIR}"

# ==============================================================================
# Setup Python virtual environment
# ==============================================================================
echo "[2/5] Setting up Python virtual environment..."
if [ ! -f "${ENV_DIR}/bin/activate" ]; then
    python3 -m venv "${ENV_DIR}"
fi

source "${ENV_DIR}/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ==============================================================================
# Install PyTorch with CUDA 12.4
# ==============================================================================
echo "[3/5] Installing PyTorch with CUDA 12.4..."
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ==============================================================================
# Install requirements
# ==============================================================================
echo "[4/5] Installing requirements..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
    pip install -r "${SCRIPT_DIR}/requirements.txt"
else
    # Install core packages if requirements.txt not found
    pip install \
        numpy==1.26.4 \
        diffusers==0.31.0 \
        transformers==4.51.3 \
        accelerate==1.12.0 \
        safetensors==0.7.0 \
        huggingface-hub==0.36.0 \
        opencv-python==4.11.0.86 \
        Pillow==12.0.0 \
        imageio==2.37.2 \
        imageio-ffmpeg==0.6.0 \
        decord==0.6.0 \
        librosa==0.11.0 \
        soundfile==0.13.1 \
        scipy==1.16.3 \
        einops==0.8.1 \
        omegaconf==2.3.0 \
        easydict==1.13 \
        tqdm==4.67.1 \
        sentencepiece==0.2.1 \
        ftfy==6.3.1 \
        regex==2025.11.3 \
        dashscope==1.25.5 \
        runpod==1.8.1
fi

# Install Flash Attention (may fail on some GPUs)
echo "Installing Flash Attention..."
pip install flash_attn==2.8.3 || echo "Flash Attention installation failed (optional)"

# ==============================================================================
# Clone WAN 2.2 repository
# ==============================================================================
echo "[5/5] Cloning WAN 2.2 repository..."
if [ ! -d "${WORKSPACE}/Wan2.2" ]; then
    git clone --depth 1 "${WAN_REPO}" "${WORKSPACE}/Wan2.2"
else
    echo "WAN 2.2 repository already exists, skipping..."
fi

# ==============================================================================
# Download models
# ==============================================================================
download_model() {
    local name=$1
    local hf_id=$2
    local dest="${MODELS_DIR}/${name}"

    if [ -d "${dest}" ] && [ "$(ls -A ${dest})" ]; then
        echo "Model ${name} already exists, skipping..."
        return 0
    fi

    echo "Downloading ${name} from ${hf_id}..."
    mkdir -p "${dest}"

    # Use huggingface-cli to download
    huggingface-cli download "${hf_id}" --local-dir "${dest}" --local-dir-use-symlinks False

    echo "Downloaded ${name} successfully!"
}

echo ""
echo "=========================================="
echo "Downloading Models"
echo "=========================================="

# Ask which models to download
echo ""
echo "Available models:"
echo "  1) Wan2.2-T2V-A14B   - Text-to-Video (14B, ~50GB)"
echo "  2) Wan2.2-I2V-A14B   - Image-to-Video (14B, ~50GB)"
echo "  3) Wan2.2-TI2V-5B    - Text+Image-to-Video (5B, ~33GB)"
echo "  4) Wan2.2-S2V-14B    - Speech-to-Video (14B, ~43GB)"
echo "  5) Wan2.2-Animate-14B - Character Animation (14B, ~50GB)"
echo "  a) All models"
echo ""

read -p "Which models to download? (e.g., '1 2' or 'a' for all): " choices

if [[ "$choices" == "a" || "$choices" == "A" ]]; then
    for name in "${!MODELS[@]}"; do
        download_model "$name" "${MODELS[$name]}"
    done
else
    for choice in $choices; do
        case $choice in
            1) download_model "Wan2.2-T2V-A14B" "Wan-AI/Wan2.2-T2V-A14B" ;;
            2) download_model "Wan2.2-I2V-A14B" "Wan-AI/Wan2.2-I2V-A14B" ;;
            3) download_model "Wan2.2-TI2V-5B" "Wan-AI/Wan2.2-TI2V-5B" ;;
            4) download_model "Wan2.2-S2V-14B" "Wan-AI/Wan2.2-S2V-14B" ;;
            5) download_model "Wan2.2-Animate-14B" "Wan-AI/Wan2.2-Animate-14B" ;;
            *) echo "Invalid choice: $choice" ;;
        esac
    done
fi

# ==============================================================================
# Verify installation
# ==============================================================================
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python -c "
import torch
import torchvision
import numpy as np
import decord

print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'NumPy: {np.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print('All imports successful!')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Models directory: ${MODELS_DIR}"
echo "Python env: ${ENV_DIR}"
echo "WAN 2.2 repo: ${WORKSPACE}/Wan2.2"
echo ""
echo "To activate the environment:"
echo "  source ${ENV_DIR}/bin/activate"
echo ""
