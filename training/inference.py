"""
Inference utilities for Nepali TTS
Load fine-tuned model and generate speech
"""
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent))


class NepaliTTS:
    """Nepali Text-to-Speech using fine-tuned Swarlekha model."""
    
    def __init__(
        self,
        weights_dir: str = "swarlekha_model/weights",
        checkpoint_dir: str = "training/outputs/checkpoints/checkpoint_35000",
        tokenizer_path: str = "training/outputs/nepali_tokenizer.json",
        device: str = "cuda",
    ):
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer_path = Path(tokenizer_path)
        
        self._load_models()
    
    def _load_models(self):
        """Load all models."""
        import torch.nn as nn
        from training.train_tokenizer import NeTokenizer
        from swarlekha_model.models.t3 import T3
        from swarlekha_model.models.s3gen import S3Gen
        from swarlekha_model.models.voice_encoder import VoiceEncoder
        
        print("Loading models...")
        
        # Load VoiceEncoder
        self.ve = VoiceEncoder()
        self.ve.load_state_dict(load_file(self.weights_dir / "ve.safetensors"))
        self.ve.to(self.device).eval()
        
        # Load S3Gen
        self.s3gen = S3Gen()
        self.s3gen.load_state_dict(
            load_file(self.weights_dir / "s3gen.safetensors"), strict=False
        )
        self.s3gen.to(self.device).eval()
        
        # Load Nepali tokenizer
        self.tokenizer = NeTokenizer(str(self.tokenizer_path))
        
        # Load T3 base model
        self.t3 = T3()
        t3_state = load_file(self.weights_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        self.t3.load_state_dict(t3_state)
        
        # Update text embedding for Nepali vocabulary
        old_vocab_size = self.t3.hp.text_tokens_dict_size
        new_vocab_size = len(self.tokenizer)
        
        if new_vocab_size != old_vocab_size:
            self.t3.text_emb = nn.Embedding(new_vocab_size, self.t3.dim)
            self.t3.text_head = nn.Linear(self.t3.dim, new_vocab_size, bias=False)
            self.t3.hp.text_tokens_dict_size = new_vocab_size
        
        # Load fine-tuned weights
        if self.checkpoint_dir.exists():
            print(f"Loading fine-tuned weights from {self.checkpoint_dir}")
            
            # Load custom weights (text embedding + head)
            custom_path = self.checkpoint_dir / "custom_weights.safetensors"
            if custom_path.exists():
                custom_state = load_file(custom_path)
                self.t3.text_emb.weight.data = custom_state['text_emb.weight'].to(self.device)
                self.t3.text_head.weight.data = custom_state['text_head.weight'].to(self.device)
            
            # Load LoRA weights
            lora_path = self.checkpoint_dir / "lora"
            if lora_path.exists():
                try:
                    from peft import PeftModel
                    self.t3.tfmr = PeftModel.from_pretrained(
                        self.t3.tfmr, 
                        str(lora_path),
                        is_trainable=False,
                    )
                    print("LoRA weights loaded")
                except Exception as e:
                    print(f"Warning: Could not load LoRA weights: {e}")
        else:
            print("Warning: No fine-tuned checkpoint found, using base model")
        
        self.t3.to(self.device).eval()
        
        # Sample rate
        from swarlekha_model.models.s3gen import S3GEN_SR
        self.sr = S3GEN_SR
        
        # Conditionals storage
        self.conds = None
        
        print("Models loaded successfully!")
    
    def prepare_voice(self, audio_path: str, exaggeration: float = 0.5):
        """Prepare voice conditioning from reference audio."""
        import librosa
        from swarlekha_model.models.s3gen import S3GEN_SR
        from swarlekha_model.models.s3tokenizer import S3_SR
        from swarlekha_model.models.t3.modules.cond_enc import T3Cond
        
        # Load audio
        s3gen_ref_wav, _ = librosa.load(audio_path, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        
        # Limit lengths
        ENC_COND_LEN = 6 * S3_SR
        DEC_COND_LEN = 10 * S3GEN_SR
        
        s3gen_ref_wav = s3gen_ref_wav[:DEC_COND_LEN]
        ref_16k_wav = ref_16k_wav[:ENC_COND_LEN]
        
        # S3Gen conditioning
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)
        
        # T3 conditioning
        plen = self.t3.hp.speech_cond_prompt_len
        t3_cond_prompt_tokens, _ = self.s3gen.tokenizer.forward(
            [ref_16k_wav], max_len=plen
        )
        t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)
        
        # Voice encoder embedding
        ve_embed = torch.from_numpy(
            self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
        ).mean(axis=0, keepdim=True).to(self.device)
        
        # Create T3 conditioning
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        
        self.conds = {
            't3': t3_cond,
            'gen': s3gen_ref_dict,
        }
        
        print(f"Voice prepared from: {audio_path}")
    
    def generate(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        cfg_weight: float = 0.5,
        repetition_penalty: float = 1.2,
    ) -> np.ndarray:
        """Generate speech from Nepali text."""
        import torch.nn.functional as F
        from swarlekha_model.models.s3tokenizer import drop_invalid_tokens, SPEECH_VOCAB_SIZE
        
        # Prepare voice if needed
        if audio_prompt_path:
            self.prepare_voice(audio_prompt_path, exaggeration)
        
        assert self.conds is not None, "Please provide audio_prompt_path or call prepare_voice() first"
        
        # Normalize text
        text = self._normalize_text(text)
        
        # Tokenize
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)
        
        # Add start/end tokens
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        
        # Duplicate for CFG
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        
        # Generate speech tokens
        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds['t3'],
                text_tokens=text_tokens,
                max_new_tokens=500,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
            )
            
            # Extract conditional batch
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < SPEECH_VOCAB_SIZE]
            speech_tokens = speech_tokens.to(self.device)
            
            # Generate audio
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds['gen'],
            )
            wav = wav.squeeze(0).cpu().numpy()
        
        return wav
    
    def _normalize_text(self, text: str) -> str:
        """Normalize Nepali text."""
        if not text or len(text.strip()) == 0:
            return "केही पाठ लेख्नुहोस्।"
        
        text = text.strip()
        text = ' '.join(text.split())
        
        # Normalize punctuation
        text = text.replace('।।', '।')
        text = text.replace('|', '।')
        
        # Add ending punctuation
        if not text.endswith(('।', '?', '!', ',')):
            text += '।'
        
        return text
    
    def save_audio(self, wav: np.ndarray, output_path: str):
        """Save audio to file."""
        import soundfile as sf
        sf.write(output_path, wav, self.sr)
        print(f"Audio saved to: {output_path}")


def main():
    """Test the Nepali TTS model."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Nepali TTS Inference")
    parser.add_argument("--text", type=str, required=True, help="Nepali text to synthesize")
    parser.add_argument("--voice", type=str, required=True, help="Path to reference voice audio")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize TTS
    tts = NepaliTTS(device=args.device)
    
    # Generate audio
    wav = tts.generate(
        text=args.text,
        audio_prompt_path=args.voice,
    )
    
    # Save
    tts.save_audio(wav, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
