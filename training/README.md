# Nepali TTS Training

This folder contains the complete training pipeline for fine-tuning the Swarlekha TTS model for Nepali language.

## Hardware Requirements

- **GPU**: RTX 3050 6GB VRAM (or better)
- **RAM**: 16GB+ recommended
- **Storage**: ~50GB for dataset and checkpoints

## Training Strategy

Due to limited VRAM (6GB), we use:

1. **LoRA (Low-Rank Adaptation)** - Only trains ~0.1% of parameters
2. **Gradient Checkpointing** - Trades compute for memory
3. **Mixed Precision (FP16)** - Halves memory usage
4. **Small Batch Size + Gradient Accumulation** - Effective batch size of 16

## Files

| File | Description |
|------|-------------|
| `config.py` | Training configuration |
| `data_prep.py` | Dataset preparation and caching |
| `train_tokenizer.py` | Train BPE tokenizer for Nepali |
| `dataset.py` | PyTorch Dataset and DataLoader |
| `train_t3.py` | Main T3 model training script |
| `inference.py` | Inference utilities for trained model |
| `run_training.sh` | One-click training script |

### Training dataset
Put the nepali open slr dataset at root: nepali_tts_dataset/

## Quick Start

### 1. Prepare Environment

```bash
# Install dependencies
pip install peft tokenizers soundfile librosa torchaudio safetensors tqdm

# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Run Training

Option A: One-click training
```bash
cd training
chmod +x run_training.sh
./run_training.sh
```

Option B: Step-by-step
```bash
# Step 1: Prepare dataset
python -m training.data_prep

# Step 2: Train tokenizer
python -m training.train_tokenizer

# Step 3: Train T3 model
python -m training.train_t3
```

### 3. Test the Model

```bash
python -m training.inference \
    --text "नमस्ते, म नेपाली बोल्छु।" \
    --voice examples/input/reference_voice.wav \
    --output output.wav
```

## Configuration

Edit `config.py` to adjust:

```python
@dataclass 
class T3TrainingConfig:
    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 8          # Increase for better quality (uses more VRAM)
    
    # Training
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    max_steps: int = 50000
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"
```

## Expected Training Time

| GPU | Estimated Time |
|-----|----------------|
| RTX 3050 6GB | 12-24 hours |
| RTX 3060 12GB | 8-16 hours |
| RTX 3090 24GB | 4-8 hours |

## Checkpoints

Checkpoints are saved to `training/outputs/checkpoints/`:
- `best/` - Best validation loss
- `checkpoint_N/` - Regular checkpoints
- `final/` - Final model

## Resuming Training

Training automatically resumes from the latest checkpoint if interrupted.

## Monitoring

Training logs are printed to console. For wandb logging:

```python
# In config.py
use_wandb: bool = True
```

## Memory Optimization Tips

If you run out of VRAM:

1. Reduce `lora_rank` (4 instead of 8)
2. Reduce `max_speech_tokens` in dataset
3. Reduce `speech_cond_prompt_len` (100 instead of 150)
4. Use `gradient_accumulation_steps = 32`

## Model Architecture

```
Pipeline:
Text → NeTokenizer → T3 (LoRA) → Speech Tokens → S3Gen → Audio
                       ↑
              VoiceEncoder (frozen)
              S3Tokenizer (frozen)
```

Only T3's attention layers are fine-tuned via LoRA. The text embedding layer is fully trained to handle Nepali vocabulary.
