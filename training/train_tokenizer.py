"""
Nepali Tokenizer Training
Extends the existing English tokenizer with Nepali (Devanagari) support.

Strategy:
  - Keep ALL existing 704 tokens (IDs 0-703) intact so pretrained T3 weights
    remain compatible.
  - Replace unused [PLACEHOLDER*] tokens (695-703) with a [ne] language tag
    and reserve the rest.
  - Train a temporary BPE tokenizer on Nepali text to learn good subword merges.
  - Append the Nepali characters + BPE merge tokens after ID 703.
  - Save the extended tokenizer for use with the T3 model.
"""
import copy
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

from tokenizers import Tokenizer, models, trainers, pre_tokenizers


# ── Constants matching the original tokenizer ──────────────────────────────────
SPACE = "[SPACE]"   # Original tokenizer's space marker (ID 2)
SOT   = "[START]"   # Start-of-text token  (ID 255)
EOT   = "[STOP]"    # End-of-text  token   (ID 0)
UNK   = "[UNK]"     # Unknown token        (ID 1)

# Devanagari Unicode ranges used in Nepali
DEVANAGARI_VOWELS = "अआइईउऊऋएऐओऔ"
DEVANAGARI_CONSONANTS = "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"
DEVANAGARI_MATRAS = "ािीुूेैोौंःँृॅॆ"
DEVANAGARI_OTHER = "्ऽॐ"
DEVANAGARI_DIGITS = "०१२३४५६७८९"
NEPALI_PUNCTUATION = "।॥"

ALL_DEVANAGARI = (
    DEVANAGARI_VOWELS + DEVANAGARI_CONSONANTS + DEVANAGARI_MATRAS +
    DEVANAGARI_OTHER + DEVANAGARI_DIGITS + NEPALI_PUNCTUATION
)


def _collect_nepali_chars(texts: List[str]) -> List[str]:
    """Collect all unique Devanagari characters that actually appear in the data."""
    chars = set()
    for text in texts:
        for ch in text:
            if '\u0900' <= ch <= '\u097F' or ch in '।॥':
                chars.add(ch)
    # Always include the base set even if not in data
    for ch in ALL_DEVANAGARI:
        chars.add(ch)
    return sorted(chars)


def _train_nepali_bpe(
    texts: List[str],
    nepali_chars: List[str],
    target_merges: int = 300,
    min_frequency: int = 5,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Train a temporary BPE tokenizer on Nepali text to learn good merges.
    
    Returns:
        vocab_tokens: Ordered list of new tokens created by BPE merges
        merges: List of (token_a, token_b) merge pairs
    """
    print(f"Training temporary Nepali BPE ({target_merges} merges)...")

    # Build a temporary tokenizer with only Nepali characters as initial alphabet
    tmp_tok = Tokenizer(models.BPE(unk_token=UNK))
    tmp_tok.pre_tokenizer = pre_tokenizers.Whitespace()

    # Replace spaces with [SPACE] in texts (matching original convention)
    processed_texts = [t.replace(' ', ' ') for t in texts]  # keep as-is for BPE

    # Train with Devanagari chars as the initial alphabet
    trainer = trainers.BpeTrainer(
        vocab_size=len(nepali_chars) + target_merges + 5,  # chars + merges + specials
        special_tokens=[UNK],
        min_frequency=min_frequency,
        show_progress=True,
        initial_alphabet=nepali_chars,
    )
    tmp_tok.train_from_iterator(processed_texts, trainer)

    # Extract the learned vocabulary and merges
    tmp_vocab = tmp_tok.get_vocab()
    # Separate: individual chars vs merged tokens
    single_chars = set(nepali_chars) | {UNK}
    merge_tokens = []
    for token in sorted(tmp_vocab.keys(), key=lambda t: tmp_vocab[t]):
        if token not in single_chars and len(token) > 1:
            merge_tokens.append(token)

    # Extract merge rules from the temporary tokenizer JSON
    tmp_json = json.loads(tmp_tok.to_str())
    raw_merges = tmp_json.get("model", {}).get("merges", [])
    merges = []
    for m in raw_merges:
        if isinstance(m, list):
            # Some tokenizers versions return merges as [token_a, token_b]
            if len(m) == 2:
                merges.append((m[0], m[1]))
        elif isinstance(m, str):
            parts = m.split(" ", 1)
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

    print(f"  Learned {len(merge_tokens)} merge tokens from {len(merges)} merge rules")
    return merge_tokens, merges


def extend_tokenizer(
    base_tokenizer_path: str,
    texts: List[str],
    output_path: str,
    nepali_merges: int = 300,
    min_frequency: int = 5,
) -> Tokenizer:
    """Extend the existing English tokenizer with Nepali tokens.
    
    Args:
        base_tokenizer_path: Path to the original tokenizer.json
        texts: List of Nepali text strings for learning BPE merges
        output_path: Path to save the extended tokenizer
        nepali_merges: Number of BPE merges to learn for Nepali
        min_frequency: Minimum frequency for BPE merges
    
    Returns:
        Extended tokenizer
    """
    print("=" * 60)
    print("Extending tokenizer with Nepali support")
    print("=" * 60)

    # ── 1. Load existing tokenizer JSON ────────────────────────────────────────
    with open(base_tokenizer_path, 'r', encoding='utf-8') as f:
        tok_data = json.load(f)

    base_vocab: Dict[str, int] = tok_data["model"]["vocab"]
    base_merges: List[str] = tok_data["model"].get("merges", [])
    added_tokens = tok_data.get("added_tokens", [])

    original_size = len(base_vocab)
    print(f"\nBase tokenizer: {original_size} tokens (IDs 0-{original_size - 1})")
    print(f"Base merges: {len(base_merges)}")

    # ── 2. Replace placeholder tokens with [ne] language tag ───────────────────
    placeholder_ids = {
        t["id"]: t for t in added_tokens
        if t["content"].startswith("[PLACEHOLDER")
    }
    ne_tag_assigned = False
    for pid in sorted(placeholder_ids.keys()):
        token_entry = placeholder_ids[pid]
        if not ne_tag_assigned:
            # First placeholder → [ne]
            old_content = token_entry["content"]
            token_entry["content"] = "[ne]"
            # Update in vocab
            base_vocab["[ne]"] = pid
            if old_content in base_vocab:
                del base_vocab[old_content]
            ne_tag_assigned = True
            print(f"  Replaced {old_content} (ID {pid}) → [ne]")
        # Keep remaining placeholders as-is (reserved for future langs)

    # ── 3. Collect Nepali characters from data ─────────────────────────────────
    nepali_chars = _collect_nepali_chars(texts)
    print(f"\nDevanagari characters found: {len(nepali_chars)}")
    print(f"  Sample: {''.join(nepali_chars[:20])}...")

    # ── 4. Train BPE merges on Nepali text ─────────────────────────────────────
    merge_tokens, merge_pairs = _train_nepali_bpe(
        texts, nepali_chars,
        target_merges=nepali_merges,
        min_frequency=min_frequency,
    )

    # ── 5. Add Nepali characters to vocabulary ─────────────────────────────────
    next_id = max(base_vocab.values()) + 1
    chars_added = 0
    for ch in nepali_chars:
        if ch not in base_vocab:
            base_vocab[ch] = next_id
            next_id += 1
            chars_added += 1
    print(f"\nAdded {chars_added} Devanagari characters (IDs {next_id - chars_added}-{next_id - 1})")

    # ── 6. Add BPE merge tokens to vocabulary ──────────────────────────────────
    merges_added = 0
    for token in merge_tokens:
        if token not in base_vocab:
            base_vocab[token] = next_id
            next_id += 1
            merges_added += 1
    print(f"Added {merges_added} BPE merge tokens (IDs {next_id - merges_added}-{next_id - 1})")

    # ── 7. Append Nepali merge rules ───────────────────────────────────────────
    existing_merges_set = set()
    for m in base_merges:
        if isinstance(m, list):
            existing_merges_set.add(f"{m[0]} {m[1]}")
        else:
            existing_merges_set.add(m)

    new_merge_count = 0
    for a, b in merge_pairs:
        merge_str = f"{a} {b}"
        if merge_str not in existing_merges_set:
            base_merges.append(merge_str)
            existing_merges_set.add(merge_str)
            new_merge_count += 1
    print(f"Added {new_merge_count} new merge rules")

    # ── 8. Update tokenizer JSON ───────────────────────────────────────────────
    tok_data["model"]["vocab"] = base_vocab
    tok_data["model"]["merges"] = base_merges

    new_size = len(base_vocab)
    print(f"\nFinal vocabulary: {new_size} tokens (was {original_size})")
    print(f"Total merges: {len(base_merges)}")

    # ── 9. Save extended tokenizer ─────────────────────────────────────────────
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tok_data, f, ensure_ascii=False, indent=2)
    print(f"\nExtended tokenizer saved to {output_path}")

    # ── 10. Also backup and save to weights dir ────────────────────────────────
    weights_dir = Path(base_tokenizer_path).parent
    backup_path = weights_dir / "tokenizer_original_en.json"
    if not backup_path.exists():
        shutil.copy2(base_tokenizer_path, backup_path)
        print(f"Original tokenizer backed up to {backup_path}")

    # ── 11. Verify by loading and testing ──────────────────────────────────────
    tokenizer = Tokenizer.from_file(str(output_path))
    _test_extended_tokenizer(tokenizer)

    return tokenizer


def _test_extended_tokenizer(tokenizer: Tokenizer):
    """Test the extended tokenizer with Nepali and English text."""
    print("\n" + "=" * 60)
    print("Tokenization Tests")
    print("=" * 60)

    test_cases = [
        # Nepali
        ("Nepali", "नमस्ते"),
        ("Nepali", "म नेपाली बोल्छु।"),
        ("Nepali", "आज मौसम राम्रो छ।"),
        ("Nepali", "काठमाडौं नेपालको राजधानी हो।"),
        # English (must still work)
        ("English", "Hello, how are you?"),
        ("English", "This is a test."),
    ]

    for lang, text in test_cases:
        # Mimic EnTokenizer: replace spaces with [SPACE]
        encoded_text = text.replace(' ', SPACE)
        encoded = tokenizer.encode(encoded_text)

        # Decode back
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        decoded = decoded.replace(' ', '').replace(SPACE, ' ').replace(EOT, '').replace(UNK, '')

        print(f"\n[{lang}] '{text}'")
        print(f"  Tokens ({len(encoded.ids)}): {encoded.tokens}")
        print(f"  IDs:    {encoded.ids}")
        print(f"  Decoded: '{decoded.strip()}'")


class NeTokenizer:
    """Extended tokenizer wrapper for Nepali + English TTS.
    
    Compatible with EnTokenizer interface: uses [SPACE] for spaces,
    [START]/[STOP] for sequence boundaries.
    """

    def __init__(self, vocab_file_path: str):
        self.tokenizer = Tokenizer.from_file(vocab_file_path)
        self.vocab_size = self.tokenizer.get_vocab_size()

        # Special token IDs (matching original EnTokenizer)
        voc = self.tokenizer.get_vocab()
        self.stop_id = voc.get(EOT, 0)    # [STOP] = 0
        self.unk_id  = voc.get(UNK, 1)    # [UNK]  = 1
        self.space_id = voc.get(SPACE, 2)  # [SPACE]= 2
        self.start_id = voc.get(SOT, 255)  # [START]= 255
        self.ne_id = voc.get("[ne]", None)

        # Aliases for compatibility
        self.pad_id = self.stop_id  # no separate pad; reuse stop
        self.bos_id = self.start_id
        self.eos_id = self.stop_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs (same convention as EnTokenizer)."""
        text = text.replace(' ', SPACE)
        encoded = self.tokenizer.encode(text)
        ids = encoded.ids
        return ids

    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        """Encode batch of texts."""
        processed = [t.replace(' ', SPACE) for t in texts]
        encoded = self.tokenizer.encode_batch(processed)
        return [e.ids for e in encoded]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        import torch
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy().tolist()
        txt = self.tokenizer.decode(ids, skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        if skip_special_tokens:
            txt = txt.replace(EOT, '').replace(SOT, '').replace(UNK, '')
        return txt.strip()

    def text_to_tokens(self, text: str) -> 'torch.Tensor':
        """Convert text to tensor of token IDs (without START/STOP)."""
        import torch
        ids = self.encode(text)
        return torch.LongTensor(ids).unsqueeze(0)

    def __len__(self):
        return self.vocab_size


def main():
    """Train extended tokenizer from prepared dataset."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from training.config import TrainingConfig
    from training.data_prep import prepare_dataset, collect_texts_for_tokenizer

    config = TrainingConfig()

    # Prepare dataset (uses cached splits if available)
    train_samples, val_samples = prepare_dataset(config)

    # Collect texts
    texts = collect_texts_for_tokenizer(train_samples)
    print(f"Collected {len(texts)} texts for tokenizer training")

    # Path to the original English tokenizer
    paths = config.get_paths()
    base_tokenizer_path = str(paths["weights"] / "tokenizer.json")

    # Extend the tokenizer
    tokenizer = extend_tokenizer(
        base_tokenizer_path=base_tokenizer_path,
        texts=texts,
        output_path=config.tokenizer.output_path,
        nepali_merges=config.tokenizer.vocab_size,  # number of BPE merges
        min_frequency=config.tokenizer.min_frequency,
    )

    print("\n" + "=" * 60)
    print("Tokenizer training complete!")
    print(f"Extended tokenizer saved to: {config.tokenizer.output_path}")
    print(f"Total vocab size: {tokenizer.get_vocab_size()}")
    print("=" * 60)

    return tokenizer


if __name__ == "__main__":
    main()
