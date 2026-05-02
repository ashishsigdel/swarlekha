#!/bin/bash
# Master script to run the complete Nepali TTS training pipeline
# Optimized for RTX 3050 6GB VRAM

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo "  Nepali TTS Training Pipeline"
echo "  Optimized for RTX 3050 6GB VRAM"
echo "=============================================="
echo ""

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Step 1: Installing dependencies..."
pip install -q peft tokenizers soundfile librosa torchaudio safetensors tqdm

echo ""
echo "Step 2: Preparing dataset..."
python -m training.data_prep

echo ""
echo "Step 3: Training Nepali tokenizer..."
python -m training.train_tokenizer

echo ""
echo "Step 4: Training T3 model with LoRA..."
echo "This will take several hours. Press Ctrl+C to stop and resume later."
echo ""
python -m training.train_t3

echo ""
echo "=============================================="
echo "  Training Complete!"
echo "=============================================="
echo ""
echo "To test the model, run:"
echo "  python -m training.inference --text 'नमस्ते' --voice path/to/voice.wav --output output.wav"
