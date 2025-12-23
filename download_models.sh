#!/bin/bash
# WAN 2.2 Model Download Script
# Downloads latest WAN 2.2 models from Hugging Face

MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR"

echo "=== WAN 2.2 Model Downloader ==="
echo "Models will be downloaded to: $MODEL_DIR"
echo ""

# Function to download model
download_model() {
    local model_name=$1
    local local_dir="$MODEL_DIR/$model_name"

    if [ -d "$local_dir" ] && [ "$(ls -A $local_dir 2>/dev/null)" ]; then
        echo "✓ $model_name already exists, skipping..."
        return 0
    fi

    echo "→ Downloading $model_name..."
    huggingface-cli download "Wan-AI/$model_name" --local-dir "$local_dir"

    if [ $? -eq 0 ]; then
        echo "✓ $model_name downloaded successfully"
    else
        echo "✗ Failed to download $model_name"
        return 1
    fi
}

# Parse arguments
MODELS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            MODELS="all"
            shift
            ;;
        --t2v)
            MODELS="$MODELS t2v"
            shift
            ;;
        --t2v-5b)
            MODELS="$MODELS t2v-5b"
            shift
            ;;
        --i2v)
            MODELS="$MODELS i2v"
            shift
            ;;
        --v2v|--animate)
            MODELS="$MODELS animate"
            shift
            ;;
        --s2v)
            MODELS="$MODELS s2v"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "WAN 2.2 Models (Latest - Apache 2.0 License):"
            echo "  --all        Download all WAN 2.2 models"
            echo "  --t2v        Download Wan2.2-T2V-A14B (Text-to-Video, 14B MoE)"
            echo "  --t2v-5b     Download Wan2.2-TI2V-5B (Text+Image-to-Video, 5B - Consumer GPU)"
            echo "  --i2v        Download Wan2.2-I2V-A14B (Image-to-Video, 14B)"
            echo "  --v2v        Download Wan2.2-Animate-14B (Video-to-Video/Animation)"
            echo "  --s2v        Download Wan2.2-S2V-14B (Speech-to-Video)"
            echo ""
            echo "Environment:"
            echo "  MODEL_DIR    Set download directory (default: ./models)"
            echo ""
            echo "Examples:"
            echo "  $0 --t2v --i2v              # Download T2V and I2V models"
            echo "  $0 --t2v-5b                 # Download lightweight 5B model (24GB VRAM)"
            echo "  MODEL_DIR=/data/models $0 --all"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no models specified, show help
if [ -z "$MODELS" ]; then
    echo "No models specified. Use --help for options."
    echo ""
    echo "Quick start:"
    echo "  $0 --t2v-5b    # Download 5B model for consumer GPUs (~10GB)"
    echo "  $0 --t2v       # Download full T2V-A14B model (~30GB)"
    echo "  $0 --all       # Download all models"
    exit 1
fi

# Download requested models
if [[ "$MODELS" == "all" ]] || [[ "$MODELS" == *"t2v"* ]] && [[ "$MODELS" != *"t2v-5b"* ]]; then
    download_model "Wan2.2-T2V-A14B"
fi

if [[ "$MODELS" == "all" ]] || [[ "$MODELS" == *"t2v-5b"* ]]; then
    download_model "Wan2.2-TI2V-5B"
fi

if [[ "$MODELS" == "all" ]] || [[ "$MODELS" == *"i2v"* ]]; then
    download_model "Wan2.2-I2V-A14B"
fi

if [[ "$MODELS" == "all" ]] || [[ "$MODELS" == *"animate"* ]]; then
    download_model "Wan2.2-Animate-14B"
fi

if [[ "$MODELS" == "all" ]] || [[ "$MODELS" == *"s2v"* ]]; then
    download_model "Wan2.2-S2V-14B"
fi

echo ""
echo "=== Download Complete ==="
echo "Models are located in: $MODEL_DIR"
