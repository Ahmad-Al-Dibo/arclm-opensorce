from __future__ import annotations

"""
Data preparation utilities for language-model training.
"""

import random
import re
import json
import os
import torch
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from .dataset import create_dataloader
from .tokenizer import SentencePieceTokenizer, Tokenizer
from .config import Config
from typing import Optional
# from .tokenizer import Tokenizer



from dataclasses import dataclass

@dataclass
class DataBundle:
    """Prepared tokens, tokenizer, and loaders used by training."""

    tokens: list
    train_tokens: list
    val_tokens: list
    # Betekent: Tokenizer, SentencePieceTokenizer, OF None
    tokenizer: Tokenizer | SentencePieceTokenizer | None
    train_loader: object
    val_loader: object
    train_encoded: list
    val_encoded: list

    @property
    def vocab_size(self):
        # Voorkom een crash als tokenizer None is
        return self.tokenizer.get_vocab_size() if self.tokenizer else 0

    def count(self):
        if self.train_encoded is None:
            raise RuntimeError(
                "train_encoded has not been initialized. Load and encode data first."
            )

        return Counter(self.tokenizer.decode(self.train_encoded))

    def save_tokenizer(self, path):
        tdata = self.tokenizer.to_json()
        with open(path, "w") as f:
            f.write(json.dumps(tdata))
        return tdata



########## Old this do not append the \n in tokens list ##########
# def read_tokens(path, limit):
#     """Read lowercase whitespace tokens from a text file."""
#     collected = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             collected.extend(line.lower().split())
#             if len(collected) >= limit:
#                 return collected[:limit]
#     return collected

def read_tokens(path, limit):
    """Read lowercase whitespace tokens from a text file, keeping newlines."""
    collected = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # 1. Controleer of de regel daadwerkelijk eindigt op een newline
            has_newline = line.endswith('\n')
            
            # 2. Splits de woorden van de regel (dit haalt de \n weg)
            words = line.lower().split()
            collected.extend(words)
            
            # 3. Voeg handmatig de '\n' toe als de regel die oorspronkelijk had
            if has_newline:
                collected.append('\n')
                
            # 4. Check direct of we de limiet al hebben bereikt
            if len(collected) >= limit:
                return collected[:limit]
                
    return collected


def load_tokens(config):
    """Load the configured dataset, optionally mixing repeated domain data first."""
    data_path = Path(config.data_path)
    if not data_path.exists():
        fallback_path = Path("data/data.txt")
        print(f"[WARNING] Data file not found: {data_path}")
        print(f"[WARNING] Falling back to {fallback_path}")
        data_path = fallback_path

    tokens = []
    domain_path = Path(config.domain_data_path) if config.domain_data_path else None
    if domain_path and domain_path.exists():
        domain_tokens = read_tokens(domain_path, config.max_data_size)
        for _ in range(config.domain_data_repeats):
            tokens.extend(domain_tokens)
            if len(tokens) >= config.max_data_size:
                return tokens[: config.max_data_size]

    remaining = config.max_data_size - len(tokens)
    if remaining > 0:
        tokens.extend(read_tokens(data_path, remaining))

    return tokens[: config.max_data_size]


def split_train_val(tokens, validation_split, block_size, seed=42):
    """Split tokens by shuffled sentences when possible, otherwise by suffix."""
    if validation_split <= 0:
        return tokens, []

    text = " ".join(tokens)
    sentences = [
        sentence.strip()
        for sentence in re.findall(r"[^.!?\n]+[.!?]?", text)
        if sentence.strip()
    ]
    if len(sentences) >= 4:
        rng = random.Random(seed)
        rng.shuffle(sentences)
        val_sentence_count = max(1, int(len(sentences) * validation_split))
        val_sentences = sentences[:val_sentence_count]
        train_sentences = sentences[val_sentence_count:]
        train_tokens = " ".join(train_sentences).split()
        val_tokens = " ".join(val_sentences).split()
        if len(train_tokens) > block_size and len(val_tokens) > block_size:
            return train_tokens, val_tokens

    min_val_tokens = block_size + 1
    val_count = max(min_val_tokens, int(len(tokens) * validation_split))

    if len(tokens) <= min_val_tokens * 2:
        return tokens, []

    val_count = min(val_count, len(tokens) - min_val_tokens)
    return tokens[:-val_count], tokens[-val_count:]


def load_tokenizer_from_checkpoint(checkpoint):
    tok_data = checkpoint.get("tokenizer") or checkpoint.get("tokenizer_metadata")

    if tok_data is None:
        return None

    if tok_data["tokenizer_type"] == "word":
        if tok_data.get("vocab") is not None:
            return Tokenizer.from_json(tok_data)
        vocab = checkpoint.get("vocab")
        if vocab is None:
            return None
        tokenizer = Tokenizer(max_vocab=tok_data.get("max_vocab", len(vocab)))
        tokenizer.vocab = vocab
        tokenizer.vocab_size = len(vocab)
        tokenizer.stoi = {token: idx for idx, token in enumerate(vocab)}
        tokenizer.itos = {idx: token for idx, token in enumerate(vocab)}
        return tokenizer

    if tok_data["tokenizer_type"] == "sentencepiece":
        if "model_proto" in tok_data and isinstance(tok_data["model_proto"], bytes):
            return SentencePieceTokenizer.from_checkpoint(tok_data)
        return SentencePieceTokenizer.from_json(tok_data)

    raise ValueError(
        f"Unknown tokenizer type: {tok_data['tokenizer_type']}"
    )


def prepare_data(config: Config, existing_tokenizer=None) -> DataBundle:
    """
    Load tokens, create train/validation splits, build tokenizer, and loaders.

    @config: Config model
    @existing_tokenizer: Optional tokenizer to use instead of creating a new one.
    @data type of config: Config
    data type of existing_tokenizer: Tokenizer | SentencePieceTokenizer | None

    return: 
          (DataBundle) model met de volgende variabels:
            tokens: list
            train_tokens: list
            val_tokens: list
            tokenizer: Tokenizer | SentencePieceTokenizer | None
            train_loader: object
            val_loader: object
            train_encoded: list
            val_encoded: list

    
    """
    tokens = load_tokens(config)
    train_tokens, val_tokens = split_train_val(
        tokens,
        config.validation_split,
        config.block_size,
        config.seed,
    )


    if existing_tokenizer is not None:
        tokenizer = existing_tokenizer

    else:
        tokenizer_type = getattr(
            config,
            "tokenizer_type",
            "word"
        )

        if tokenizer_type == "sentencepiece":
            tokenizer = SentencePieceTokenizer(
                max_vocab=config.max_vocab,
                model_type=getattr(
                    config,
                    "sentencepiece_model_type",
                    "bpe"
                ),
                character_coverage=getattr(
                    config,
                    "sentencepiece_character_coverage",
                    1.0
                ),
                user_defined_symbols=getattr(
                    config,
                    "user_defined_symbols",
                    None
                ),
            )

        elif tokenizer_type == "word":
            tokenizer = Tokenizer(
                max_vocab=config.max_vocab,
                user_defined_symbols=getattr(
                    config,
                    "user_defined_symbols",
                    None
                ),
            )

        else:
            raise ValueError(
                f"Unsupported tokenizer_type: {tokenizer_type}"
            )

        tokenizer.build(" ".join(train_tokens))

    train_encoded = tokenizer.encode(train_tokens)
    train_loader = create_dataloader(
        train_encoded,
        config.block_size,
        config.batch_size,
        shuffle=True,
    )

    val_encoded = []
    val_loader = None
    if val_tokens:
        val_encoded = tokenizer.encode(val_tokens)
        val_loader = create_dataloader(
            val_encoded,
            config.block_size,
            config.batch_size,
            shuffle=False,
        )

    return DataBundle(
        tokens=tokens,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        train_encoded=train_encoded,
        val_encoded=val_encoded,
    )
