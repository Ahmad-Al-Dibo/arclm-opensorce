"""ArcLM-owned tokenizer facade and internal tokenizer engines."""

from __future__ import annotations

import base64
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from .templates import ChatFormatter, ChatTemplate


class TokenizerEngine(Protocol):
    """Internal contract implemented by tokenizer strategies."""

    name: str

    @property
    def vocab_size(self) -> int: ...
    @property
    def is_built(self) -> bool: ...
    def build(self, text: str) -> None: ...
    def tokenize(self, text: str) -> list[str]: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...


class Tokenizer:
    """Single public tokenizer API for ArcLM workflows."""

    DEFAULT_SPECIAL_TOKENS = {
        "bos": "<s>",
        "eos": "</s>",
        "pad": "<pad>",
        "unk": "<UNK>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
        "system": "<|system|>",
        "tool": "<|tool|>",
        "separator": "<|sep|>",
    }
    _engines: dict[str, type[TokenizerEngine]] = {}
    _aliases = {
        "default": "word",
        "words": "word",
        "char": "character",
        "chars": "character",
        "sentencepiece": "sentence",
        "sp": "sentence",
        "bpe": "sentence",
        "unigram": "sentence",
    }

    def __init__(
        self,
        strategy: str = "word",
        *,
        special_tokens: dict[str, str] | None = None,
        chat_template: ChatTemplate | ChatFormatter | dict[str, Any] | str | None = None,
        add_default_special_tokens: bool = True,
        **options: Any,
    ):
        normalized = self.normalize_strategy(strategy)
        engine_cls = self._engines.get(normalized)
        if engine_cls is None:
            supported = "', '".join(sorted(self._engines))
            raise ValueError(f"Unknown tokenizer strategy: {strategy!r}. Supported: '{supported}'.")
        self.strategy = normalized
        self.special_tokens = self._resolve_special_tokens(special_tokens, add_default=add_default_special_tokens)
        self.chat_template = self._resolve_chat_template(chat_template)
        user_defined = list(options.pop("user_defined_symbols", []) or [])
        for token in self.special_tokens.values():
            if token and token not in user_defined and token != self.special_tokens.get("unk"):
                user_defined.append(token)
        if user_defined:
            options["user_defined_symbols"] = user_defined
        if "unknown_token" not in options and self.special_tokens.get("unk"):
            options["unknown_token"] = self.special_tokens["unk"]
        self.engine = engine_cls(**options)

    @classmethod
    def register_engine(cls, name: str, engine_cls: type[TokenizerEngine], *, aliases: tuple[str, ...] = ()) -> None:
        normalized = cls.normalize_strategy(name)
        cls._engines[normalized] = engine_cls
        for alias in aliases:
            cls._aliases[alias.lower().strip()] = normalized

    @classmethod
    def normalize_strategy(cls, strategy: str | None) -> str:
        key = str(strategy or "word").lower().strip().replace("_", "-")
        return cls._aliases.get(key, key)

    @property
    def vocab_size(self) -> int:
        return self.engine.vocab_size

    @property
    def is_built(self) -> bool:
        return self.engine.is_built

    def build(self, text: str) -> "Tokenizer":
        self.engine.build(text)
        return self

    def tokenize(self, text: str) -> list[str]:
        return self._tokenize_with_specials(str(text))

    def encode(self, text: str | list[str], *, add_special_tokens: bool = False) -> list[int]:
        if isinstance(text, list):
            text = " ".join(str(token) for token in text if token)
        value = str(text)
        if add_special_tokens:
            value = self._with_boundary_tokens(value)
        if hasattr(self.engine, "stoi"):
            if not self.is_built:
                raise ValueError("Tokenizer is not built. Call build() first.")
            unknown = self.get_unknown_index()
            return [int(self.engine.stoi.get(token, unknown)) for token in self.tokenize(value)]  # type: ignore[attr-defined]
        return self.engine.encode(value)

    def decode(self, token_ids: list[int]) -> str:
        return self.engine.decode([int(token_id) for token_id in token_ids])

    def encode_text(self, text: str) -> list[int]:
        """Compatibility alias inside the current implementation."""

        return self.encode(text)

    def decode_string(self, token_ids: list[int]) -> str:
        """Compatibility alias inside the current implementation."""

        return self.decode(token_ids)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_unknown_index(self) -> int:
        return int(getattr(self.engine, "unknown_index", 0))

    def special_token(self, name: str) -> str:
        try:
            return self.special_tokens[str(name).lower().strip()]
        except KeyError as exc:
            raise KeyError(f"Unknown special token: {name!r}.") from exc

    def special_token_id(self, name: str) -> int:
        token = self.special_token(name)
        if not self.is_built:
            raise ValueError("Tokenizer is not built. Call build() first.")
        if hasattr(self.engine, "stoi"):
            return int(self.engine.stoi[token])  # type: ignore[attr-defined]
        encoded = self.engine.encode(token)
        if len(encoded) != 1:
            raise ValueError(f"Special token {name!r} does not map to exactly one token id.")
        return int(encoded[0])

    def format_chat(self, messages: list[dict[str, Any]], *, add_generation_prompt: bool = False, template: ChatTemplate | ChatFormatter | None = None) -> str:
        active = template or self.chat_template or ChatTemplate.default()
        return active.format(messages, add_generation_prompt=add_generation_prompt)

    def to_json(self) -> dict[str, Any]:
        payload = self.engine.to_dict()
        payload["format"] = "arclm-tokenizer"
        payload["schema_version"] = "1"
        payload["strategy"] = self.strategy
        payload["vocab_size"] = self.vocab_size
        payload["special_tokens"] = dict(self.special_tokens)
        if isinstance(self.chat_template, ChatTemplate):
            payload["chat_template"] = self.chat_template.to_dict()
        return payload

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Tokenizer":
        strategy = cls.normalize_strategy(data.get("strategy") or data.get("tokenizer_type") or "word")
        if strategy == "transformers-compat":
            from ..compat.transformers import TransformersTokenizerAdapter

            return TransformersTokenizerAdapter.from_json(data)  # type: ignore[return-value]
        engine_cls = cls._engines.get(strategy)
        if engine_cls is None:
            raise ValueError(f"Unsupported tokenizer strategy in artifact: {strategy!r}.")
        tokenizer = cls.__new__(cls)
        tokenizer.strategy = strategy
        tokenizer.special_tokens = cls._resolve_special_tokens(data.get("special_tokens"), add_default=True)
        tokenizer.chat_template = cls._resolve_chat_template(data.get("chat_template"))
        tokenizer.engine = engine_cls.from_dict(data)  # type: ignore[attr-defined]
        return tokenizer

    def summary(self) -> None:
        """Print a tokenizer summary."""

        print(f"Tokenizer strategy: {self.strategy}")
        print(f"Vocab size: {self.vocab_size}")
        print(f"Total tokens: {sum(self.engine.token_counts.values())}")
        print(f"Is built: {self.is_built}")

    @classmethod
    def _resolve_special_tokens(cls, special_tokens: dict[str, str] | None, *, add_default: bool) -> dict[str, str]:
        tokens = dict(cls.DEFAULT_SPECIAL_TOKENS if add_default else {})
        for name, token in dict(special_tokens or {}).items():
            if token is not None:
                tokens[str(name).lower().strip()] = str(token)
        return tokens

    @staticmethod
    def _resolve_chat_template(chat_template: ChatTemplate | ChatFormatter | dict[str, Any] | str | None) -> ChatTemplate | ChatFormatter:
        if chat_template is None:
            return ChatTemplate.default()
        if isinstance(chat_template, str):
            return ChatTemplate.get(chat_template)
        if isinstance(chat_template, dict):
            return ChatTemplate.from_dict(chat_template)
        return chat_template

    def _tokenize_with_specials(self, text: str) -> list[str]:
        special_values = sorted({token for token in self.special_tokens.values() if token}, key=len, reverse=True)
        if not special_values:
            return self.engine.tokenize(text)
        pattern = "(" + "|".join(re.escape(token) for token in special_values) + ")"
        tokens: list[str] = []
        for part in re.split(pattern, text):
            if not part:
                continue
            if part in self.special_tokens.values():
                tokens.append(part)
            else:
                tokens.extend(self.engine.tokenize(part))
        return tokens

    def _with_boundary_tokens(self, text: str) -> str:
        parts = []
        if self.special_tokens.get("bos"):
            parts.append(self.special_tokens["bos"])
        parts.append(str(text))
        if self.special_tokens.get("eos"):
            parts.append(self.special_tokens["eos"])
        return " ".join(part for part in parts if part)


class WordTokenizerEngine:
    """Whitespace tokenizer engine."""

    name = "word"

    def __init__(self, max_vocab: int = 50000, unknown_token: str = "<UNK>", user_defined_symbols: list[str] | None = None):
        self.max_vocab = int(max_vocab)
        self.unknown_token = unknown_token
        self.user_defined_symbols = list(user_defined_symbols or [])
        self.vocab: list[str] = []
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.token_counts: Counter[str] = Counter()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def is_built(self) -> bool:
        return bool(self.stoi)

    @property
    def unknown_index(self) -> int:
        return self.stoi.get(self.unknown_token, 0)

    def build(self, text: str) -> None:
        tokens = self.tokenize(text)
        self.token_counts = Counter(tokens)
        reserved = []
        for token in self.user_defined_symbols:
            if token and token != self.unknown_token and token not in reserved:
                reserved.append(token)
        remaining = max(self.max_vocab - 1 - len(reserved), 0)
        most_common = [
            token
            for token, _ in self.token_counts.most_common()
            if token != self.unknown_token and token not in reserved
        ][:remaining]
        self.vocab = [self.unknown_token] + reserved[: max(self.max_vocab - 1, 0)] + most_common
        self.stoi = {token: index for index, token in enumerate(self.vocab)}
        self.itos = {index: token for token, index in self.stoi.items()}

    def tokenize(self, text: str) -> list[str]:
        return [token for token in str(text).split() if token]

    def encode(self, text: str) -> list[int]:
        if not self.is_built:
            raise ValueError("Tokenizer is not built. Call build() first.")
        return [self.stoi.get(token, self.unknown_index) for token in self.tokenize(text)]

    def decode(self, token_ids: list[int]) -> str:
        if not self.is_built:
            raise ValueError("Tokenizer is not built. Call build() first.")
        return " ".join(self.itos[token_id] for token_id in token_ids if token_id in self.itos)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_vocab": self.max_vocab,
            "unknown_token": self.unknown_token,
            "user_defined_symbols": list(self.user_defined_symbols),
            "vocab": list(self.vocab),
            "token_counts": dict(self.token_counts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WordTokenizerEngine":
        engine = cls(
            max_vocab=int(data.get("max_vocab", 50000)),
            unknown_token=data.get("unknown_token") or data.get("default_token", "<UNK>"),
            user_defined_symbols=data.get("user_defined_symbols"),
        )
        engine.vocab = list(data.get("vocab") or [])
        engine.stoi = {token: index for index, token in enumerate(engine.vocab)}
        engine.itos = {index: token for token, index in engine.stoi.items()}
        engine.token_counts = Counter(data.get("token_counts", {}))
        return engine


class CharacterTokenizerEngine(WordTokenizerEngine):
    """Character-level tokenizer engine."""

    name = "character"

    def tokenize(self, text: str) -> list[str]:
        return list(str(text))

    def encode(self, text: str) -> list[int]:
        if not self.is_built:
            raise ValueError("Tokenizer is not built. Call build() first.")
        return [self.stoi.get(token, self.unknown_index) for token in self.tokenize(text)]

    def decode(self, token_ids: list[int]) -> str:
        if not self.is_built:
            raise ValueError("Tokenizer is not built. Call build() first.")
        return "".join(self.itos[token_id] for token_id in token_ids if token_id in self.itos)


class SentencePieceTokenizerEngine:
    """Optional SentencePiece-backed internal engine."""

    name = "sentence"

    def __init__(
        self,
        max_vocab: int = 50000,
        model_type: str = "bpe",
        character_coverage: float = 1.0,
        user_defined_symbols: list[str] | None = None,
    ):
        self.max_vocab = int(max_vocab)
        self.model_type = model_type
        self.character_coverage = float(character_coverage)
        self.user_defined_symbols = list(user_defined_symbols or [])
        self.processor = self._sentencepiece().SentencePieceProcessor()
        self.token_counts: Counter[str] = Counter()

    @staticmethod
    def _sentencepiece():
        try:
            import sentencepiece as spm
        except Exception as exc:  # pragma: no cover - only exercised without optional dep
            raise RuntimeError("Tokenizer(strategy='sentence') requires sentencepiece.") from exc
        return spm

    @property
    def vocab_size(self) -> int:
        return int(self.processor.get_piece_size())

    @property
    def is_built(self) -> bool:
        return self.vocab_size > 0

    @property
    def unknown_index(self) -> int:
        return int(self.processor.unk_id())

    def build(self, text: str) -> None:
        self.token_counts = Counter(str(text).split())
        spm = self._sentencepiece()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "tokenizer-input.txt"
            model_prefix = tmp / "tokenizer"
            input_path.write_text(self._format_training_text(text), encoding="utf-8")
            spm.SentencePieceTrainer.train(
                input=str(input_path),
                model_prefix=str(model_prefix),
                vocab_size=self.max_vocab,
                model_type=self.model_type,
                character_coverage=self.character_coverage,
                user_defined_symbols=[token for token in self.user_defined_symbols if token],
                hard_vocab_limit=False,
                unk_id=0,
                bos_id=-1,
                eos_id=-1,
                pad_id=-1,
            )
            self.processor.load(str(model_prefix) + ".model")

    def tokenize(self, text: str) -> list[str]:
        if not self.is_built:
            raise ValueError("Tokenizer is not built. Call build() first.")
        return list(self.processor.encode(str(text), out_type=str))

    def encode(self, text: str) -> list[int]:
        if not self.is_built:
            raise ValueError("Tokenizer is not built. Call build() first.")
        return [int(token_id) for token_id in self.processor.encode(str(text), out_type=int)]

    def decode(self, token_ids: list[int]) -> str:
        if not self.is_built:
            raise ValueError("Tokenizer is not built. Call build() first.")
        return str(self.processor.decode([int(token_id) for token_id in token_ids]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_vocab": self.max_vocab,
            "model_type": self.model_type,
            "character_coverage": self.character_coverage,
            "user_defined_symbols": list(self.user_defined_symbols),
            "model_proto": base64.b64encode(self.processor.serialized_model_proto()).decode("utf-8"),
            "token_counts": dict(self.token_counts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentencePieceTokenizerEngine":
        engine = cls(
            max_vocab=int(data.get("max_vocab", 50000)),
            model_type=data.get("model_type", "bpe"),
            character_coverage=float(data.get("character_coverage", 1.0)),
            user_defined_symbols=data.get("user_defined_symbols"),
        )
        model_proto = data.get("model_proto")
        if not model_proto:
            raise ValueError("Sentence tokenizer artifact is missing model_proto.")
        engine.processor.load_from_serialized_proto(base64.b64decode(model_proto))
        engine.token_counts = Counter(data.get("token_counts", {}))
        return engine

    @staticmethod
    def _format_training_text(text: str, max_line_length: int = 4000) -> str:
        lines: list[str] = []
        for raw_line in str(text).splitlines() or [str(text)]:
            words = raw_line.split()
            current: list[str] = []
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


Tokenizer.register_engine("word", WordTokenizerEngine, aliases=("default", "words"))
Tokenizer.register_engine("character", CharacterTokenizerEngine, aliases=("char", "chars"))
Tokenizer.register_engine("sentence", SentencePieceTokenizerEngine, aliases=("sentencepiece", "sp", "bpe", "unigram"))

__all__ = ["ChatTemplate", "Tokenizer", "TokenizerEngine"]
