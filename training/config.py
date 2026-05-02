"""
Training Configuration for Nepali TTS
Optimized for RTX 3050 6GB VRAM
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Paths
    dataset_dir: str = "nepali_tts_dataset"
    tsv_file: str = "utt_spk_text.tsv"
    data_subdir: str = "data"
    
    # Output paths
    output_dir: str = "training/outputs"
    cache_dir: str = "training/cache"
    
    # Data processing
    max_audio_len: float = 15.0  # seconds
    min_audio_len: float = 0.5   # seconds
    max_text_len: int = 200      # characters
    
    # Train/val split
    val_ratio: float = 0.05
    seed: int = 42


@dataclass
class TokenizerConfig:
    """Nepali tokenizer configuration (extends existing English tokenizer)"""
    vocab_size: int = 512         # Number of BPE merges to learn for Nepali
    min_frequency: int = 5        # Minimum frequency for BPE merge pairs
    output_path: str = "training/outputs/nepali_tokenizer.json"


@dataclass 
class T3TrainingConfig:
    """T3 Model training configuration - Optimized for 6GB VRAM"""
    # Pretrained weights
    pretrained_dir: str = "swarlekha_model/weights"
    
    # LoRA configuration (essential for low VRAM)
    use_lora: bool = True
    lora_rank: int = 8          # Lower rank = less memory
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    
    # Training hyperparameters
    batch_size: int = 1         # Minimal batch size for 6GB
    gradient_accumulation_steps: int = 16  # Effective batch = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50000
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: str = "fp16"  # Use fp16 for memory savings
    max_grad_norm: float = 1.0
    
    # Scheduler
    lr_scheduler: str = "cosine"
    
    # Logging & saving
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000
    max_checkpoints: int = 3
    
    # Inference settings (for generation during eval)
    max_speech_tokens: int = 500
    temperature: float = 0.8
    cfg_weight: float = 0.5


@dataclass
class S3GenTrainingConfig:
    """S3Gen training configuration (optional, for fine-tuning vocoder)"""
    # Generally not needed, but included for completeness
    pretrained_path: str = "swarlekha_model/weights/s3gen.safetensors"
    
    # Only train flow matching, freeze tokenizer and speaker encoder
    train_flow: bool = True
    train_vocoder: bool = False
    
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    max_steps: int = 20000


@dataclass
class TrainingConfig:
    """Master training configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    t3: T3TrainingConfig = field(default_factory=T3TrainingConfig)
    s3gen: S3GenTrainingConfig = field(default_factory=S3GenTrainingConfig)
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 2  # Keep low to save memory
    pin_memory: bool = True
    
    # Experiment tracking
    project_name: str = "swarlekha-nepali"
    experiment_name: Optional[str] = None
    use_wandb: bool = False  # Set True if you want wandb logging
    
    def get_paths(self):
        """Get absolute paths based on workspace root"""
        import os
        workspace = Path(__file__).parent.parent.absolute()
        return {
            "workspace": workspace,
            "dataset": workspace / self.data.dataset_dir,
            "output": workspace / self.data.output_dir,
            "cache": workspace / self.data.cache_dir,
            "weights": workspace / self.t3.pretrained_dir,
        }
