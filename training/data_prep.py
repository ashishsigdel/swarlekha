"""
Data Preparation Module for Nepali TTS Training
Handles dataset loading, preprocessing, and caching
"""
import os
import csv
import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import re

import torch
import torchaudio
import numpy as np


# Nepali Unicode range
NEPALI_PATTERN = re.compile(r'[\u0900-\u097F\s\।\॥0-9०-९\,\.\?\!\-\'\"\(\)]+')


@dataclass
class Sample:
    """Single data sample"""
    utterance_id: str
    speaker_id: str
    text: str
    audio_path: str
    duration: Optional[float] = None


def clean_nepali_text(text: str) -> str:
    """Clean and normalize Nepali text."""
    if not text or len(text.strip()) == 0:
        return ""
    
    text = text.strip()
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    # Normalize punctuation
    text = text.replace('।।', '।')
    text = text.replace('|', '।')
    
    # Ensure ending punctuation
    if text and not text.endswith(('।', '?', '!', ',')):
        text += '।'
    
    return text


def get_audio_path(dataset_dir: Path, utterance_id: str) -> Path:
    """Get audio file path from utterance ID.
    
    Audio files are stored in subdirectories based on first 2 chars of ID.
    e.g., utterance_id='4aa1fdca33' -> data/4a/4aa1fdca33.flac
    """
    subdir = utterance_id[:2]
    return dataset_dir / "data" / subdir / f"{utterance_id}.flac"


def load_dataset_metadata(dataset_dir: Path, tsv_file: str = "utt_spk_text.tsv") -> List[Sample]:
    """Load dataset metadata from TSV file."""
    samples = []
    tsv_path = dataset_dir / tsv_file
    
    print(f"Loading metadata from {tsv_path}...")
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading TSV")):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            utterance_id = parts[0].strip()
            speaker_id = parts[1].strip()
            text = parts[2].strip()
            
            # Clean text
            text = clean_nepali_text(text)
            if not text or len(text) < 2:
                continue
            
            # Get audio path
            audio_path = get_audio_path(dataset_dir, utterance_id)
            if not audio_path.exists():
                continue
            
            samples.append(Sample(
                utterance_id=utterance_id,
                speaker_id=speaker_id,
                text=text,
                audio_path=str(audio_path),
            ))
    
    print(f"Loaded {len(samples)} valid samples")
    return samples


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        info = torchaudio.info(audio_path)
        return info.num_frames / info.sample_rate
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return -1


def filter_by_duration(
    samples: List[Sample],
    min_duration: float = 0.5,
    max_duration: float = 15.0,
    cache_path: Optional[Path] = None,
) -> List[Sample]:
    """Filter samples by audio duration with caching."""
    
    # Try to load from cache
    if cache_path and cache_path.exists():
        print(f"Loading duration cache from {cache_path}")
        with open(cache_path, 'r') as f:
            duration_cache = json.load(f)
    else:
        duration_cache = {}
    
    filtered = []
    new_durations = {}
    
    for sample in tqdm(samples, desc="Filtering by duration"):
        # Check cache first
        if sample.utterance_id in duration_cache:
            duration = duration_cache[sample.utterance_id]
        else:
            duration = get_audio_duration(sample.audio_path)
            new_durations[sample.utterance_id] = duration
        
        if min_duration <= duration <= max_duration:
            sample.duration = duration
            filtered.append(sample)
    
    # Update cache
    if cache_path and new_durations:
        duration_cache.update(new_durations)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(duration_cache, f)
        print(f"Updated duration cache with {len(new_durations)} new entries")
    
    print(f"Filtered to {len(filtered)} samples (duration: {min_duration}-{max_duration}s)")
    return filtered


def split_dataset(
    samples: List[Sample],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample]]:
    """Split dataset into train and validation sets."""
    random.seed(seed)
    
    # Shuffle samples
    samples = samples.copy()
    random.shuffle(samples)
    
    # Split
    val_size = int(len(samples) * val_ratio)
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    
    print(f"Split: {len(train_samples)} train, {len(val_samples)} val")
    return train_samples, val_samples


def get_speaker_stats(samples: List[Sample]) -> Dict[str, int]:
    """Get speaker statistics."""
    speaker_counts = {}
    for sample in samples:
        speaker_counts[sample.speaker_id] = speaker_counts.get(sample.speaker_id, 0) + 1
    return speaker_counts


def save_samples_to_json(samples: List[Sample], output_path: Path):
    """Save samples to JSON file."""
    data = [
        {
            "utterance_id": s.utterance_id,
            "speaker_id": s.speaker_id,
            "text": s.text,
            "audio_path": s.audio_path,
            "duration": s.duration,
        }
        for s in samples
    ]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} samples to {output_path}")


def load_samples_from_json(json_path: Path) -> List[Sample]:
    """Load samples from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = [
        Sample(
            utterance_id=d["utterance_id"],
            speaker_id=d["speaker_id"],
            text=d["text"],
            audio_path=d["audio_path"],
            duration=d.get("duration"),
        )
        for d in data
    ]
    return samples


def prepare_dataset(config) -> Tuple[List[Sample], List[Sample]]:
    """Main function to prepare the dataset."""
    paths = config.get_paths()
    dataset_dir = paths["dataset"]
    cache_dir = paths["cache"]
    output_dir = paths["output"]
    
    # Check for cached splits
    train_json = output_dir / "train_samples.json"
    val_json = output_dir / "val_samples.json"
    
    if train_json.exists() and val_json.exists():
        print("Loading cached train/val splits...")
        train_samples = load_samples_from_json(train_json)
        val_samples = load_samples_from_json(val_json)
        return train_samples, val_samples
    
    # Load metadata
    samples = load_dataset_metadata(dataset_dir, config.data.tsv_file)
    
    # Filter by duration
    duration_cache = cache_dir / "duration_cache.json"
    samples = filter_by_duration(
        samples,
        min_duration=config.data.min_audio_len,
        max_duration=config.data.max_audio_len,
        cache_path=duration_cache,
    )
    
    # Filter by text length
    samples = [s for s in samples if len(s.text) <= config.data.max_text_len]
    print(f"After text length filter: {len(samples)} samples")
    
    # Split
    train_samples, val_samples = split_dataset(
        samples,
        val_ratio=config.data.val_ratio,
        seed=config.data.seed,
    )
    
    # Save splits
    save_samples_to_json(train_samples, train_json)
    save_samples_to_json(val_samples, val_json)
    
    # Print stats
    train_speakers = get_speaker_stats(train_samples)
    print(f"Training speakers: {len(train_speakers)}")
    print(f"Total training duration: {sum(s.duration for s in train_samples) / 3600:.2f} hours")
    
    return train_samples, val_samples


def collect_texts_for_tokenizer(samples: List[Sample]) -> List[str]:
    """Collect all texts for tokenizer training."""
    return [s.text for s in samples]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from training.config import TrainingConfig
    
    config = TrainingConfig()
    train_samples, val_samples = prepare_dataset(config)
    
    print(f"\n=== Dataset Summary ===")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Sample texts
    print(f"\nSample texts:")
    for i in range(min(5, len(train_samples))):
        print(f"  {i+1}. {train_samples[i].text}")
