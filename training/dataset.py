"""
PyTorch Dataset for Nepali TTS Training
"""
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np

from training.data_prep import Sample


# Constants from the model
S3_SR = 16000       # S3Tokenizer sample rate
S3GEN_SR = 24000    # S3Gen sample rate
S3_TOKEN_RATE = 25  # tokens per second


class NepaliTTSDataset(Dataset):
    """Dataset for Nepali TTS training."""
    
    def __init__(
        self,
        samples: List[Sample],
        tokenizer,
        s3_tokenizer,
        voice_encoder,
        max_text_tokens: int = 200,
        max_speech_tokens: int = 500,
        speech_cond_prompt_len: int = 150,
        device: str = "cuda",
        cache_embeddings: bool = True,
    ):
        """
        Args:
            samples: List of Sample objects
            tokenizer: Nepali text tokenizer
            s3_tokenizer: S3Tokenizer for speech tokens
            voice_encoder: VoiceEncoder for speaker embeddings
            max_text_tokens: Maximum text token length
            max_speech_tokens: Maximum speech token length
            speech_cond_prompt_len: Length of speech conditioning prompt
            device: Device for processing
            cache_embeddings: Whether to cache embeddings (saves compute but uses more RAM)
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.s3_tokenizer = s3_tokenizer
        self.voice_encoder = voice_encoder
        self.max_text_tokens = max_text_tokens
        self.max_speech_tokens = max_speech_tokens
        self.speech_cond_prompt_len = speech_cond_prompt_len
        self.device = device
        self.cache_embeddings = cache_embeddings
        
        # Cache for speaker embeddings (optional)
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._speech_token_cache: Dict[str, torch.Tensor] = {}
        
    def __len__(self):
        return len(self.samples)
    
    def _load_audio(self, audio_path: str, target_sr: int) -> torch.Tensor:
        """Load and resample audio."""
        wav, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)
        
        # Resample if needed
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        
        return wav
    
    @torch.no_grad()
    def _get_speech_tokens(self, wav_16k: torch.Tensor, utterance_id: str) -> torch.Tensor:
        """Get speech tokens using S3Tokenizer."""
        # Check cache
        if self.cache_embeddings and utterance_id in self._speech_token_cache:
            return self._speech_token_cache[utterance_id]
        
        # Tokenize
        wav_input = wav_16k.unsqueeze(0).to(self.device)
        speech_tokens, _ = self.s3_tokenizer.forward(wav_input)
        speech_tokens = speech_tokens.squeeze(0).cpu()
        
        # Cache
        if self.cache_embeddings:
            self._speech_token_cache[utterance_id] = speech_tokens
        
        return speech_tokens
    
    @torch.no_grad()
    def _get_speaker_embedding(self, wav_16k: np.ndarray, speaker_id: str) -> torch.Tensor:
        """Get speaker embedding using VoiceEncoder."""
        # Check cache
        if self.cache_embeddings and speaker_id in self._embedding_cache:
            return self._embedding_cache[speaker_id]
        
        # Get embedding
        embedding = self.voice_encoder.embeds_from_wavs([wav_16k], sample_rate=S3_SR)
        embedding = torch.from_numpy(embedding).squeeze(0)
        
        # Cache
        if self.cache_embeddings:
            self._embedding_cache[speaker_id] = embedding
        
        return embedding
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load audio at 16kHz
        wav_16k = self._load_audio(sample.audio_path, S3_SR)
        wav_16k_np = wav_16k.numpy()
        
        # Tokenize text
        text_tokens = self.tokenizer.encode(sample.text, add_special_tokens=False)
        text_tokens = torch.LongTensor(text_tokens)
        
        # Truncate text if needed
        if len(text_tokens) > self.max_text_tokens:
            text_tokens = text_tokens[:self.max_text_tokens]
        
        # Get speech tokens
        speech_tokens = self._get_speech_tokens(wav_16k, sample.utterance_id)
        
        # Truncate speech tokens if needed
        if len(speech_tokens) > self.max_speech_tokens:
            speech_tokens = speech_tokens[:self.max_speech_tokens]
        
        # Get speaker embedding
        speaker_emb = self._get_speaker_embedding(wav_16k_np, sample.speaker_id)
        
        # Get conditioning prompt tokens (first few seconds of speech)
        cond_prompt_tokens = speech_tokens[:self.speech_cond_prompt_len]
        
        return {
            'text_tokens': text_tokens,
            'speech_tokens': speech_tokens,
            'speaker_emb': speaker_emb,
            'cond_prompt_tokens': cond_prompt_tokens,
            'text': sample.text,
            'utterance_id': sample.utterance_id,
            'speaker_id': sample.speaker_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with dynamic padding."""
    # Find max lengths
    max_text_len = max(len(x['text_tokens']) for x in batch)
    max_speech_len = max(len(x['speech_tokens']) for x in batch)
    max_cond_len = max(len(x['cond_prompt_tokens']) for x in batch)
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    speech_tokens = torch.zeros(batch_size, max_speech_len, dtype=torch.long)
    cond_prompt_tokens = torch.zeros(batch_size, max_cond_len, dtype=torch.long)
    
    text_lens = torch.zeros(batch_size, dtype=torch.long)
    speech_lens = torch.zeros(batch_size, dtype=torch.long)
    cond_lens = torch.zeros(batch_size, dtype=torch.long)
    
    speaker_embs = []
    texts = []
    utterance_ids = []
    speaker_ids = []
    
    for i, item in enumerate(batch):
        # Text
        tl = len(item['text_tokens'])
        text_tokens[i, :tl] = item['text_tokens']
        text_lens[i] = tl
        
        # Speech
        sl = len(item['speech_tokens'])
        speech_tokens[i, :sl] = item['speech_tokens']
        speech_lens[i] = sl
        
        # Conditioning
        cl = len(item['cond_prompt_tokens'])
        cond_prompt_tokens[i, :cl] = item['cond_prompt_tokens']
        cond_lens[i] = cl
        
        # Others
        speaker_embs.append(item['speaker_emb'])
        texts.append(item['text'])
        utterance_ids.append(item['utterance_id'])
        speaker_ids.append(item['speaker_id'])
    
    speaker_embs = torch.stack(speaker_embs)
    
    return {
        'text_tokens': text_tokens,
        'text_lens': text_lens,
        'speech_tokens': speech_tokens,
        'speech_lens': speech_lens,
        'cond_prompt_tokens': cond_prompt_tokens,
        'cond_lens': cond_lens,
        'speaker_embs': speaker_embs,
        'texts': texts,
        'utterance_ids': utterance_ids,
        'speaker_ids': speaker_ids,
    }


def create_dataloader(
    samples: List[Sample],
    tokenizer,
    s3_tokenizer,
    voice_encoder,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    device: str = "cuda",
) -> DataLoader:
    """Create DataLoader for training."""
    # This dataset performs CUDA work inside __getitem__ for speech tokenization
    # and speaker embedding extraction. That must happen in the main process,
    # otherwise forked DataLoader workers will crash when they touch CUDA.
    if device == "cuda":
        num_workers = 0

    dataset = NepaliTTSDataset(
        samples=samples,
        tokenizer=tokenizer,
        s3_tokenizer=s3_tokenizer,
        voice_encoder=voice_encoder,
        device=device,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
