"""
Tokenizer Module - Text to tokens conversion
"""

import tempfile
from collections import Counter
from pathlib import Path
import base64
import json
import sentencepiece as spm


class Tokenizer:
    """Tokenizer for converting text to token indices and back"""
    
    def __init__(self, max_vocab=50000, default_token="<UNK>", user_defined_symbols=None):
        self.max_vocab = max_vocab
        self.default_token = default_token
        self.user_defined_symbols = list(user_defined_symbols or [])
        self.vocab = None
        self.stoi = None  # string to index
        self.itos = None  # index to string
        self.vocab_size = 0
        self.token_counts = Counter()
    
    def build(self, text):
        """Build vocabulary from text"""
        tokens = text.split()
        token_counts = Counter(tokens)
        self.token_counts = token_counts
        
        reserved = []
        for token in self.user_defined_symbols:
            if token and token != self.default_token and token not in reserved:
                reserved.append(token)

        remaining_vocab = max(self.max_vocab - 1 - len(reserved), 0)
        most_common = [
            token
            for token, _ in token_counts.most_common()
            if token != self.default_token and token not in reserved
        ][:remaining_vocab]
        self.vocab = [self.default_token] + reserved[: max(self.max_vocab - 1, 0)] + most_common
        
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for word, i in self.stoi.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, tokens):
        """Encode tokens to indices"""
        if self.stoi is None:
            raise ValueError("Tokenizer not built. Call build() first.")
        
        unk_idx = self.stoi.get(self.default_token, 0)
        return [self.stoi.get(token, unk_idx) for token in tokens if token]

    def encode_text(self, text):
        """Encode raw text to token indices."""
        return self.encode(text.split())
    
    def decode(self, indices):
        """Decode indices back to tokens"""
        if self.itos is None:
            raise ValueError("Tokenizer not built. Call build() first.")
        
        return [self.itos[idx] for idx in indices if idx in self.itos]
    
    def decode_string(self, indices):
        """Decode indices to string"""
        return " ".join(self.decode(indices))
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return self.vocab_size

    def get_unknown_index(self):
        """Get the index used for unknown tokens."""
        if self.stoi is None:
            raise ValueError("Tokenizer not built. Call build() first.")
        return self.stoi.get(self.default_token, 0)

    def get_rare_tokens(self, min_count=2, limit=20):
        """Return training tokens that occur fewer than min_count times."""
        if not self.token_counts:
            return []

        rare_tokens = [
            (token, count)
            for token, count in self.token_counts.items()
            if count < min_count
        ]
        rare_tokens.sort(key=lambda item: (item[1], item[0]))
        return rare_tokens[:limit]

    def analyze_coverage(self, tokens, rare_threshold=2, top_unknown=20):
        """Measure unknown-token and vocabulary coverage for a token sequence."""
        if self.stoi is None:
            raise ValueError("Tokenizer not built. Call build() first.")

        clean_tokens = [token for token in tokens if token]
        total_tokens = len(clean_tokens)
        unknown_counter = Counter(token for token in clean_tokens if token not in self.stoi)
        unknown_tokens = sum(unknown_counter.values())
        unique_tokens = set(clean_tokens)
        known_unique = {token for token in unique_tokens if token in self.stoi}
        rare_tokens = self.get_rare_tokens(min_count=rare_threshold, limit=top_unknown)

        return {
            "total_tokens": total_tokens,
            "known_tokens": total_tokens - unknown_tokens,
            "unknown_tokens": unknown_tokens,
            "unknown_rate": unknown_tokens / total_tokens if total_tokens else 0.0,
            "unique_tokens": len(unique_tokens),
            "known_unique_tokens": len(known_unique),
            "unknown_unique_tokens": len(unique_tokens) - len(known_unique),
            "vocab_size": self.vocab_size,
            "vocab_coverage": len(known_unique) / len(unique_tokens) if unique_tokens else 0.0,
            "rare_threshold": rare_threshold,
            "rare_training_tokens": rare_tokens,
            "top_unknown_tokens": unknown_counter.most_common(top_unknown),
        }
    
    def save_vocab(self, path):
        """Save vocabulary to file"""
        with open(path, "w", encoding="utf-8") as f:
            for token in self.vocab:
                f.write(token + "\n")
    
    def load_vocab(self, path):
        """Load vocabulary from file"""
        with open(path, "r", encoding="utf-8") as f:
            self.vocab = [line.strip() for line in f.readlines()]
        
        self.stoi = {word: i for i, word in enumerate(self.vocab)}
        self.itos = {i: word for word, i in self.stoi.items()}
        self.vocab_size = len(self.vocab)
        self.token_counts = Counter()

    def to_checkpoint(self):
        """Return tokenizer metadata that can be stored in a model checkpoint."""
        return {
            "tokenizer_type": "word",
            "max_vocab": self.max_vocab,
            "user_defined_symbols": self.user_defined_symbols,
        }

    def to_json(self):
        """Serialize tokenizer to JSON-compatible dict."""
        return {
            "tokenizer_type": "word",
            "max_vocab": self.max_vocab,
            "user_defined_symbols": self.user_defined_symbols,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "token_counts": dict(self.token_counts),
        }
    
    def save(self, path):
        """Save tokenizer to a JSON file."""
        metadata = self.to_json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        
        return {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "stoi": self.stoi,
            "itos": self.itos
        }

    @classmethod
    def from_json(cls, data):
        """Load a word tokenizer from JSON-compatible dict."""
        tokenizer = cls(
            max_vocab=data["max_vocab"],
            user_defined_symbols=data.get("user_defined_symbols"),
        )

        tokenizer.vocab = data["vocab"]
        tokenizer.vocab_size = data["vocab_size"]

        tokenizer.stoi = {
            token: idx
            for idx, token in enumerate(tokenizer.vocab)
        }

        tokenizer.itos = {
            idx: token
            for idx, token in enumerate(tokenizer.vocab)
        }

        tokenizer.token_counts = Counter(
            data.get("token_counts", {})
        )

        return tokenizer


class SentencePieceTokenizer:
    """SentencePiece BPE tokenizer for subword language-model training."""

    def __init__(self, max_vocab=50000, model_type="bpe", character_coverage=1.0, default_token="<UNK>", model_name=None, user_defined_symbols=None,):
        self.max_vocab = max_vocab
        self.model_type = model_type
        self.model_name = model_name
        self.character_coverage = character_coverage
        self.default_token = default_token
        self.processor = spm.SentencePieceProcessor()
        self.vocab = None
        self.stoi = None
        self.itos = None
        self.vocab_size = 0
        self.token_counts = Counter()
        self.user_defined_symbols=[
        "<|qa_start|>",
        "<|res_start|>",
        "<|end|>",
        "<|pad|>",
    ] if user_defined_symbols is None else user_defined_symbols

    def _format_training_text(self, text, max_line_length=4000):
        """Wrap training text so SentencePiece does not skip oversized lines."""
        lines = []
        for raw_line in text.splitlines() or [text]:
            words = raw_line.split()
            if not words:
                continue

            current = []
            current_length = 0
            for word in words:
                extra = len(word) + (1 if current else 0)
                if current and current_length + extra > max_line_length:
                    lines.append(" ".join(current))
                    current = [word]
                    current_length = len(word)
                else:
                    current.append(word)
                    current_length += extra

            if current:
                lines.append(" ".join(current))

        return "\n".join(lines)

    def build(self, text:str) -> dict:
        """
        Train a SentencePiece tokenizer from raw text.
        @text: text data for the trainer.

        return:
            {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "stoi": self.stoi,
            "itos": self.itos
        }
        
        """
        self.token_counts = Counter(text.split())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "sentencepiece_input.txt"
            model_prefix = tmpdir_path / "sentencepiece_tokenizer"
            input_path.write_text(self._format_training_text(text), encoding="utf-8")

            user_defined_symbols = [
                token
                for token in self.user_defined_symbols
                if token and token != "<|pad|>"
            ]

            spm.SentencePieceTrainer.train(
                input=str(input_path),
                model_prefix=str(model_prefix),
                vocab_size=self.max_vocab,
                model_type=self.model_type,
                character_coverage=self.character_coverage,
                user_defined_symbols=user_defined_symbols,
                hard_vocab_limit=False,
                unk_id=0,
                bos_id=-1,
                eos_id=-1,
                pad_id=1,
                pad_piece="<|pad|>",
            )
            self.processor.load(str(model_prefix) + ".model")

        return self._refresh_vocab()

    def to_checkpoint(self):
        """Return tokenizer metadata that can be stored in a model checkpoint."""
        return {
            "tokenizer_type": "sentencepiece",
            "max_vocab": self.max_vocab,
            "model_type": self.model_type,
            "character_coverage": self.character_coverage,
            "user_defined_symbols": self.user_defined_symbols,
            "model_proto": self.processor.serialized_model_proto(),
        }

    @classmethod
    def from_checkpoint(cls, metadata):
        """Rebuild a SentencePiece tokenizer from checkpoint metadata."""
        tokenizer = cls(
            max_vocab=metadata.get("max_vocab", 50000),
            model_type=metadata.get("model_type", "bpe"),
            character_coverage=metadata.get("character_coverage", 1.0),
            user_defined_symbols=metadata.get("user_defined_symbols"),
        )
        model_proto = metadata.get("model_proto")
        if not model_proto:
            raise ValueError("SentencePiece tokenizer metadata is missing model_proto.")
        tokenizer.processor.load_from_serialized_proto(model_proto)
        tokenizer._refresh_vocab()
        return tokenizer

    def _refresh_vocab(self):
        self.vocab_size = self.processor.get_piece_size()
        self.vocab = [self.processor.id_to_piece(idx) for idx in range(self.vocab_size)]
        self.stoi = {piece: idx for idx, piece in enumerate(self.vocab)}
        self.itos = {idx: piece for idx, piece in enumerate(self.vocab)}
        return {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "stoi": self.stoi,
            "itos": self.itos
        }

    def encode(self, tokens):
        """Encode tokens by joining them back into text first."""
        return self.encode_text(" ".join(token for token in tokens if token))

    def encode_text(self, text, out_type=int):
        """Encode raw text to SentencePiece token IDs."""
        return self.processor.encode(text, out_type=out_type)

    def decode(self, indices):
        """Decode IDs to SentencePiece pieces."""
        return [self.itos[idx] for idx in indices if idx in self.itos]

    def decode_string(self, indices):
        """Decode IDs directly back to text."""
        return self.processor.decode([int(idx) for idx in indices])

    def get_vocab_size(self):
        """Get vocabulary size."""
        return self.vocab_size

    def get_unknown_index(self):
        """Get the index used for unknown pieces."""
        return self.processor.unk_id()

    def get_rare_tokens(self, min_count=2, limit=20):
        """Return rare whitespace tokens from the training text."""
        rare_tokens = [
            (token, count)
            for token, count in self.token_counts.items()
            if count < min_count
        ]
        rare_tokens.sort(key=lambda item: (item[1], item[0]))
        return rare_tokens[:limit]
    

    def to_json(self):
        """Serialize SentencePiece tokenizer."""
        return {
            "tokenizer_type": "sentencepiece",
            "model_name": self.model_name,
            "max_vocab": self.max_vocab,
            "model_type": self.model_type,
            "character_coverage": self.character_coverage,
            "user_defined_symbols": self.user_defined_symbols,
            "vocab_size": self.vocab_size,
            "model_proto": base64.b64encode(
                self.processor.serialized_model_proto()
            ).decode("utf-8"),
            "token_counts": dict(self.token_counts),
        }

    def save(self, path):
        """Save SentencePiece tokenizer to a file."""
        metadata = self.to_json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        
        return {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "stoi": self.stoi,
            "itos": self.itos
        }

    @classmethod
    def from_json(cls, data):
        """Load SentencePiece tokenizer from JSON-compatible dict."""
        tokenizer = cls(
            max_vocab=data["max_vocab"],
            model_type=data["model_type"],
            character_coverage=data["character_coverage"],
            model_name=data.get("model_name"),
            user_defined_symbols=data.get("user_defined_symbols"),
        )
        model_proto = base64.b64decode(data["model_proto"])
        tokenizer.processor.load_from_serialized_proto(model_proto)
        tokenizer.token_counts = Counter(data.get("token_counts", {}))
        tokenizer._refresh_vocab()
        return tokenizer

    @classmethod
    def load(cls, path):

        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls(
            max_vocab=data["max_vocab"],
            model_type=data["model_type"],
            character_coverage=data["character_coverage"],
            user_defined_symbols=data.get("user_defined_symbols"),
        )
        model_proto = base64.b64decode(
            data["model_proto"]
        )
        tokenizer.processor.load_from_serialized_proto(
            model_proto
        )
        tokenizer.token_counts = Counter(
            data.get("token_counts", {})
        )
        tokenizer._refresh_vocab()
        return tokenizer

    def analyze_coverage(self, tokens, rare_threshold=2, top_unknown=20):
        """Measure unknown-piece usage for a token sequence."""
        text = " ".join(token for token in tokens if token)
        pieces = self.processor.encode(text, out_type=str)
        ids = self.processor.encode(text, out_type=int)
        unk_id = self.get_unknown_index()
        unknown_pieces = [piece for piece, idx in zip(pieces, ids) if idx == unk_id]
        unknown_counter = Counter(unknown_pieces)
        total_pieces = len(ids)

        return {
            "total_tokens": total_pieces,
            "known_tokens": total_pieces - len(unknown_pieces),
            "unknown_tokens": len(unknown_pieces),
            "unknown_rate": len(unknown_pieces) / total_pieces if total_pieces else 0.0,
            "unique_tokens": len(set(pieces)),
            "known_unique_tokens": len(set(pieces) - set(unknown_pieces)),
            "unknown_unique_tokens": len(set(unknown_pieces)),
            "vocab_size": self.vocab_size,
            "vocab_coverage": 1.0 - (
                len(set(unknown_pieces)) / len(set(pieces)) if pieces else 0.0
            ),
            "rare_threshold": rare_threshold,
            "rare_training_tokens": self.get_rare_tokens(
                min_count=rare_threshold,
                limit=top_unknown,
            ),
            "top_unknown_tokens": unknown_counter.most_common(top_unknown),
        }

    def get_meta_data(self):
        """Return metadata for the tokenizer."""
        return {
            "tokenizer_type": "sentencepiece",
            "max_vocab": self.max_vocab,
            "model_type": self.model_type,
            "character_coverage": self.character_coverage,
            "user_defined_symbols": self.user_defined_symbols,
            "vocab_size": self.vocab_size,
        }


# ============================================================================
# TOKENIZER FACTORY (P4-2: Multi-Tokenizer Support)
# ============================================================================

class TokenizerFactory:
    """Single selection point for tokenizer implementations."""

    _registry = {}

    @classmethod
    def register(cls, tokenizer_type, tokenizer_class):
        """Register a tokenizer class by type name."""
        cls._registry[tokenizer_type.lower().strip()] = tokenizer_class

    @classmethod
    def create(cls, tokenizer_type="word", **kwargs):
        """Create a tokenizer by type name."""
        tokenizer_type = tokenizer_type.lower().strip()
        if tokenizer_type not in cls._registry:
            supported = "', '".join(sorted(cls._registry))
            raise ValueError(
                f"Unknown tokenizer type: {tokenizer_type}. "
                f"Supported: '{supported}'"
            )
        return cls._registry[tokenizer_type](**kwargs)

    @classmethod
    def from_config(cls, config):
        """Create a tokenizer from a Config-like object."""
        tokenizer_type = getattr(config, 'tokenizer_type', 'word')
        max_vocab = getattr(config, 'vocab_size', None) or getattr(config, 'max_vocab', 50000)

        if tokenizer_type == "sentencepiece":
            model_type = getattr(config, 'sentencepiece_model_type', 'bpe')
            character_coverage = getattr(config, 'sentencepiece_character_coverage', 1.0)
            return cls.create(
                "sentencepiece",
                max_vocab=max_vocab,
                model_type=model_type,
                character_coverage=character_coverage,
                user_defined_symbols=getattr(config, "user_defined_symbols", None),
            )

        return cls.create(
            "word",
            max_vocab=max_vocab,
            user_defined_symbols=getattr(config, "user_defined_symbols", None),
        )


TokenizerFactory.register("word", Tokenizer)
TokenizerFactory.register("sentencepiece", SentencePieceTokenizer)

def create_tokenizer(tokenizer_type="word", **kwargs):
    """
    Factory function to create tokenizer instances.
    
    Args:
        tokenizer_type (str): "word" (default) or "sentencepiece"
        **kwargs: Additional parameters passed to the tokenizer constructor
        
    Returns:
        Tokenizer or SentencePieceTokenizer instance
        
    Examples:
        >>> tok = create_tokenizer("word", max_vocab=10000)
        >>> tok = create_tokenizer("sentencepiece", model_type="bpe")
    """
    return TokenizerFactory.create(tokenizer_type, **kwargs)


def get_tokenizer_from_config(config):
    """
    Convenience function to create tokenizer from Config object.
    
    Args:
        config: Config object with tokenizer_type and related settings
        
    Returns:
        Tokenizer instance ready to use
    """
    return TokenizerFactory.from_config(config)


