"""
T3 Model Training for Nepali TTS
Uses LoRA for efficient fine-tuning on RTX 3050 6GB VRAM
"""
import os
import sys
import gc
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import TrainingConfig
from training.data_prep import prepare_dataset, load_samples_from_json
from training.train_tokenizer import NeTokenizer
from training.dataset import create_dataloader


def setup_lora(model: nn.Module, config) -> nn.Module:
    """Apply LoRA to the T3 model for efficient fine-tuning."""
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        print("Installing peft...")
        os.system("pip install peft")
        from peft import get_peft_model, LoraConfig, TaskType
    
    # LoRA configuration optimized for low VRAM
    lora_config = LoraConfig(
        # T3 uses a bare LlamaModel backbone, not an LM head wrapper.
        # FEATURE_EXTRACTION avoids PEFT expecting generation helpers that
        # LlamaModel does not implement.
        task_type=TaskType.FEATURE_EXTRACTION,
        r=config.t3.lora_rank,
        lora_alpha=config.t3.lora_alpha,
        lora_dropout=config.t3.lora_dropout,
        target_modules=config.t3.lora_target_modules,
        bias="none",
    )
    
    # Apply LoRA to the transformer backbone
    model.tfmr = get_peft_model(model.tfmr, lora_config)
    
    print("\n=== LoRA Configuration ===")
    model.tfmr.print_trainable_parameters()
    
    return model


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_trainable_params(model):
    """Get trainable parameters."""
    return [p for p in model.parameters() if p.requires_grad]


class T3Trainer:
    """Trainer for T3 model fine-tuning."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.paths = config.get_paths()
        self.device = config.device
        
        # Initialize logging
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create output directories
        self.output_dir = self.paths["output"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup experiment name
        if config.experiment_name is None:
            self.experiment_name = f"nepali_t3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = config.experiment_name
    
    def load_models(self):
        """Load pretrained models."""
        print("\n=== Loading Models ===")
        weights_dir = self.paths["weights"]
        
        # Load VoiceEncoder
        print("Loading VoiceEncoder...")
        from swarlekha_model.models.voice_encoder import VoiceEncoder
        self.ve = VoiceEncoder()
        self.ve.load_state_dict(load_file(weights_dir / "ve.safetensors"))
        self.ve.to(self.device).eval()
        
        # Load S3Gen (only need tokenizer)
        print("Loading S3Tokenizer...")
        from swarlekha_model.models.s3gen import S3Gen
        self.s3gen = S3Gen()
        self.s3gen.load_state_dict(
            load_file(weights_dir / "s3gen.safetensors"), strict=False
        )
        self.s3gen.to(self.device).eval()
        self.s3_tokenizer = self.s3gen.tokenizer
        
        # Free unnecessary S3Gen components to save memory
        del self.s3gen.flow
        del self.s3gen.mel2wav
        clear_memory()
        
        # Load T3 model
        print("Loading T3...")
        from swarlekha_model.models.t3 import T3
        self.t3 = T3()
        t3_state = load_file(weights_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        self.t3.load_state_dict(t3_state)
        
        # Load Nepali tokenizer
        print("Loading Nepali tokenizer...")
        tokenizer_path = self.paths["output"] / "nepali_tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Nepali tokenizer not found at {tokenizer_path}. "
                "Please run: python -m training.train_tokenizer"
            )
        self.tokenizer = NeTokenizer(str(tokenizer_path))
        
        # Update T3 text embedding for new vocabulary
        self._update_text_embedding()
        
        # Apply LoRA if enabled
        if self.config.t3.use_lora:
            self.t3 = setup_lora(self.t3, self.config)
        
        # Enable gradient checkpointing
        if self.config.t3.gradient_checkpointing:
            self.t3.tfmr.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
        self.t3.to(self.device)
        print("Models loaded successfully!")
        
        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def _update_text_embedding(self):
        """Update text embedding layer for Nepali vocabulary."""
        old_vocab_size = self.t3.hp.text_tokens_dict_size
        new_vocab_size = len(self.tokenizer)
        
        if new_vocab_size != old_vocab_size:
            print(f"Updating text embedding: {old_vocab_size} -> {new_vocab_size}")
            
            old_embedding = self.t3.text_emb.weight.data.clone()
            old_head = self.t3.text_head.weight.data.clone()
            
            # Create new embedding layer
            self.t3.text_emb = nn.Embedding(new_vocab_size, self.t3.dim)
            self.t3.text_head = nn.Linear(self.t3.dim, new_vocab_size, bias=False)
            
            # Initialize with small random values
            nn.init.normal_(self.t3.text_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.t3.text_head.weight, mean=0.0, std=0.02)
            
            # Copy overlapping weights
            min_vocab = min(old_vocab_size, new_vocab_size)
            self.t3.text_emb.weight.data[:min_vocab] = old_embedding[:min_vocab]
            self.t3.text_head.weight.data[:min_vocab] = old_head[:min_vocab]
            
            self.t3.hp.text_tokens_dict_size = new_vocab_size
            print(f"Text embedding updated successfully")
    
    def load_data(self):
        """Load and prepare training data."""
        print("\n=== Loading Data ===")
        
        # Load prepared samples
        train_json = self.paths["output"] / "train_samples.json"
        val_json = self.paths["output"] / "val_samples.json"
        
        if not train_json.exists():
            print("Preparing dataset...")
            train_samples, val_samples = prepare_dataset(self.config)
        else:
            train_samples = load_samples_from_json(train_json)
            val_samples = load_samples_from_json(val_json)
        
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        
        # Create dataloaders
        self.train_loader = create_dataloader(
            samples=train_samples,
            tokenizer=self.tokenizer,
            s3_tokenizer=self.s3_tokenizer,
            voice_encoder=self.ve,
            batch_size=self.config.t3.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            device=self.device,
        )
        
        self.val_loader = create_dataloader(
            samples=val_samples[:100],  # Use subset for faster validation
            tokenizer=self.tokenizer,
            s3_tokenizer=self.s3_tokenizer,
            voice_encoder=self.ve,
            batch_size=self.config.t3.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            device=self.device,
        )
        
        print("Data loaded!")
    
    def setup_training(self):
        """Setup optimizer and scheduler."""
        print("\n=== Setting up Training ===")
        
        # Get trainable parameters
        trainable_params = get_trainable_params(self.t3)
        num_params = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameters: {num_params:,}")
        
        # Optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.t3.learning_rate,
            weight_decay=self.config.t3.weight_decay,
            betas=(0.9, 0.999),
        )
        
        # Scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.t3.warmup_steps,
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.t3.max_steps - self.config.t3.warmup_steps,
            eta_min=self.config.t3.learning_rate * 0.01,
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.t3.warmup_steps],
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.config.t3.mixed_precision == "fp16")
        
        print("Training setup complete!")
    
    def compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute training loss."""
        # Move to device
        text_tokens = batch['text_tokens'].to(self.device)
        speech_tokens = batch['speech_tokens'].to(self.device)
        speaker_embs = batch['speaker_embs'].to(self.device)
        cond_prompt_tokens = batch['cond_prompt_tokens'].to(self.device)
        text_lens = batch['text_lens'].to(self.device)
        speech_lens = batch['speech_lens'].to(self.device)
        cond_lens = batch['cond_lens'].to(self.device)
        
        batch_size = text_tokens.size(0)
        
        # Add start/end tokens to text
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        text_lens = text_lens + 2
        
        # Add start token to speech (input) and end token to target
        sos = self.t3.hp.start_speech_token
        eos = self.t3.hp.stop_speech_token
        
        # Input: [SOS, tok1, tok2, ...] (shift right)
        speech_input = F.pad(speech_tokens, (1, 0), value=sos)
        # Target: [tok1, tok2, ..., EOS]
        speech_target = F.pad(speech_tokens, (0, 1), value=eos)
        
        # Create T3 conditioning
        from swarlekha_model.models.t3.modules.cond_enc import T3Cond
        
        # Embed conditioning prompt tokens
        cond_prompt_emb = self.t3.speech_emb(cond_prompt_tokens)
        if hasattr(self.t3, 'speech_pos_emb'):
            cond_prompt_emb = cond_prompt_emb + self.t3.speech_pos_emb(cond_prompt_tokens)
        
        t3_cond = T3Cond(
            speaker_emb=speaker_embs.unsqueeze(1),
            cond_prompt_speech_tokens=cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_emb,
            emotion_adv=0.5 * torch.ones(batch_size, 1, 1, device=self.device),
        )
        
        # Forward pass
        output = self.t3.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_lens,
            speech_tokens=speech_input,
            speech_token_lens=speech_lens + 1,
            training=True,
        )
        
        # Compute loss
        speech_logits = output.speech_logits  # (B, seq_len, vocab_size)
        
        # Create mask for valid positions
        max_len = speech_target.size(1)
        mask = torch.arange(max_len, device=self.device)[None, :] < (speech_lens + 1)[:, None]
        
        # Flatten for cross-entropy
        logits_flat = speech_logits.reshape(-1, speech_logits.size(-1))
        target_flat = speech_target.reshape(-1)
        mask_flat = mask.reshape(-1)
        
        # Compute masked cross-entropy loss
        loss = F.cross_entropy(logits_flat, target_flat, reduction='none')
        loss = (loss * mask_flat.float()).sum() / mask_flat.float().sum()
        
        return loss
    
    def train_step(self, batch: Dict) -> float:
        """Single training step."""
        self.t3.train()
        
        # Forward with mixed precision
        with autocast(enabled=self.config.t3.mixed_precision == "fp16"):
            loss = self.compute_loss(batch)
            loss = loss / self.config.t3.gradient_accumulation_steps
        
        # Backward
        self.scaler.scale(loss).backward()
        
        return loss.item() * self.config.t3.gradient_accumulation_steps
    
    def optimizer_step(self):
        """Perform optimizer step."""
        # Unscale and clip gradients
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            get_trainable_params(self.t3),
            self.config.t3.max_grad_norm
        )
        
        # Step optimizer
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Step scheduler
        self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.t3.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            with autocast(enabled=self.config.t3.mixed_precision == "fp16"):
                loss = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str = None, is_best: bool = False):
        """Save model checkpoint."""
        if name is None:
            name = f"checkpoint_{self.global_step}"
        
        checkpoint_path = self.checkpoint_dir / name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save LoRA weights if using LoRA
        if self.config.t3.use_lora:
            self.t3.tfmr.save_pretrained(checkpoint_path / "lora")
        
        # Save full model state (text embedding + heads that were modified)
        custom_state = {
            'text_emb.weight': self.t3.text_emb.weight.data.cpu(),
            'text_head.weight': self.t3.text_head.weight.data.cpu(),
        }
        save_file(custom_state, checkpoint_path / "custom_weights.safetensors")
        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
        }
        torch.save(training_state, checkpoint_path / "training_state.pt")
        
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Manage checkpoints (keep only N most recent)
        if not is_best:
            checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_*"),
                key=lambda x: int(x.name.split("_")[1])
            )
            while len(checkpoints) > self.config.t3.max_checkpoints:
                oldest = checkpoints.pop(0)
                import shutil
                shutil.rmtree(oldest)
                print(f"Removed old checkpoint: {oldest}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // self.config.t3.gradient_accumulation_steps
        total_epochs = math.ceil(self.config.t3.max_steps / steps_per_epoch)
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total epochs: {total_epochs}")
        print(f"Max steps: {self.config.t3.max_steps}")
        
        # Training loop
        accumulated_loss = 0.0
        accumulated_steps = 0
        
        for epoch in range(total_epochs):
            print(f"\n=== Epoch {epoch + 1}/{total_epochs} ===")
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss
                accumulated_steps += 1
                
                # Optimizer step after accumulation
                if accumulated_steps >= self.config.t3.gradient_accumulation_steps:
                    self.optimizer_step()
                    self.global_step += 1
                    
                    avg_loss = accumulated_loss / accumulated_steps
                    accumulated_loss = 0.0
                    accumulated_steps = 0
                    
                    # Update progress bar
                    lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                        'step': self.global_step,
                    })
                    
                    # Logging
                    if self.global_step % self.config.t3.log_every == 0:
                        if torch.cuda.is_available():
                            mem = torch.cuda.memory_allocated() / 1024**3
                            print(f"\nStep {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}, mem={mem:.2f}GB")
                    
                    # Validation
                    if self.global_step % self.config.t3.eval_every == 0:
                        val_loss = self.validate()
                        print(f"\nValidation loss: {val_loss:.4f}")
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint("best", is_best=True)
                            print(f"New best model! Loss: {val_loss:.4f}")
                    
                    # Save checkpoint
                    if self.global_step % self.config.t3.save_every == 0:
                        self.save_checkpoint()
                    
                    # Check if done
                    if self.global_step >= self.config.t3.max_steps:
                        print("\nTraining complete!")
                        self.save_checkpoint("final")
                        return
        
        print("\nTraining complete!")
        self.save_checkpoint("final")
    
    def run(self):
        """Run complete training pipeline."""
        try:
            self.load_models()
            self.load_data()
            self.setup_training()
            self.train()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint("interrupted")
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    config = TrainingConfig()
    
    # Print configuration
    print("="*60)
    print("Nepali TTS Training - T3 Model")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Batch size: {config.t3.batch_size}")
    print(f"Gradient accumulation: {config.t3.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.t3.batch_size * config.t3.gradient_accumulation_steps}")
    print(f"LoRA enabled: {config.t3.use_lora}")
    print(f"LoRA rank: {config.t3.lora_rank}")
    print(f"Mixed precision: {config.t3.mixed_precision}")
    print(f"Max steps: {config.t3.max_steps}")
    
    trainer = T3Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
