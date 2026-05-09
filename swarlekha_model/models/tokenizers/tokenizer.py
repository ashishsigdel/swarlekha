import logging
from pathlib import Path
from unicodedata import normalize

import torch
from tokenizers import Tokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class EnTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str):
        text_tokens = self.encode(text)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode( self, txt: str, verbose=False):
        """
        clean_text > (append `lang_id`) > replace SPACE > encode text using Tokenizer
        """
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq,
        skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt


class NepaliTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        # Ensure [ne] is present
        new_tokens = ["[ne]"]
        self.tokenizer.add_tokens([t for t in new_tokens if t not in self.tokenizer.get_vocab()])
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def preprocess_text(self, raw_text: str, lowercase: bool = True, nfkd_normalize: bool = True):
        """
        Text preprocessor that handles lowercase conversion and NFKD normalization.
        """
        preprocessed_text = raw_text
        if lowercase:
            preprocessed_text = preprocessed_text.lower()
        if nfkd_normalize:
            preprocessed_text = normalize("NFKD", preprocessed_text)
        
        return preprocessed_text

    def text_to_tokens(self, text: str, lowercase: bool = True, nfkd_normalize: bool = True):
        text_tokens = self.encode(text, lowercase=lowercase, nfkd_normalize=nfkd_normalize)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(self, txt: str, lowercase: bool = True, nfkd_normalize: bool = True):
        txt = self.preprocess_text(txt, lowercase=lowercase, nfkd_normalize=nfkd_normalize)
        
        # prepend Nepali language token
        lang_tag = "[ne]"

        if not txt.startswith(lang_tag):
            txt = f"{lang_tag} {txt}"
        
        txt = txt.replace(' ', SPACE)
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt = self.tokenizer.decode(seq, skip_special_tokens=False)
        txt = txt.replace(' ', '').replace(SPACE, ' ').replace(EOT, '').replace(UNK, '')
        return txt
