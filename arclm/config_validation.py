"""Validation helpers shared by Python APIs and the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .exceptions import ConfigurationError


VALID_DEVICES = {"auto", "cpu", "cuda"}
VALID_PRECISIONS = {"auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"}


def normalize_device(device: Optional[str]) -> str:
    """Normalize and validate a device string."""

    value = "auto" if device is None else str(device).lower().strip()
    if value not in VALID_DEVICES:
        raise ConfigurationError(
            "device must be one of: " + ", ".join(sorted(VALID_DEVICES))
        )
    if value == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("device='cuda' was requested, but CUDA is not available.")
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def normalize_precision(precision: Optional[str]) -> str:
    """Normalize and validate a precision string."""

    value = "auto" if precision is None else str(precision).lower().replace("torch.", "").strip()
    if value not in VALID_PRECISIONS:
        raise ConfigurationError(
            "precision must be one of: " + ", ".join(sorted(VALID_PRECISIONS))
        )
    aliases = {"float": "float32", "half": "float16"}
    return aliases.get(value, value)


def normalize_path(path: Any) -> str:
    """Return a normalized filesystem path string."""

    return str(Path(path).expanduser())


def validate_training_config(config: Any) -> Any:
    """Validate important native training configuration fields in place."""

    int_positive = ["embed_dim", "block_size", "batch_size", "num_epochs", "num_blocks", "max_vocab"]
    for field in int_positive:
        value = getattr(config, field, None)
        if value is None or int(value) <= 0:
            raise ConfigurationError(f"{field} must be a positive integer.")

    for field in ("learning_rate",):
        value = getattr(config, field, None)
        if value is None or float(value) <= 0:
            raise ConfigurationError(f"{field} must be greater than zero.")

    dropout = float(getattr(config, "dropout", 0.0))
    if dropout < 0.0 or dropout >= 1.0:
        raise ConfigurationError("dropout must be >= 0.0 and < 1.0.")

    validation_split = float(getattr(config, "validation_split", 0.0))
    if validation_split < 0.0 or validation_split >= 1.0:
        raise ConfigurationError("validation_split must be >= 0.0 and < 1.0.")

    tokenizer_type = str(getattr(config, "tokenizer_type", "word")).lower().strip()
    if tokenizer_type not in {"word", "sentencepiece"}:
        raise ConfigurationError("tokenizer_type must be one of: word, sentencepiece.")
    config.tokenizer_type = tokenizer_type

    config.device = normalize_device(getattr(config, "device", "auto"))
    config.data_path = normalize_path(getattr(config, "data_path", "data/data.txt"))
    config.model_path = normalize_path(getattr(config, "model_path", "output/model.pth"))
    config.tokenizer_path = normalize_path(getattr(config, "tokenizer_path", "output/tokenizer.model"))

    seed = getattr(config, "seed", None)
    if seed is None or int(seed) < 0:
        raise ConfigurationError("seed must be a non-negative integer.")
    config.seed = int(seed)
    return config


def validated_config_dict(values: Dict[str, Any], allowed: set[str]) -> Dict[str, Any]:
    """Validate unknown config fields and return a shallow copy."""

    unknown = set(values) - set(allowed)
    if unknown:
        raise ConfigurationError(
            "Unknown configuration parameters: " + ", ".join(sorted(unknown))
        )
    return dict(values)


__all__ = [
    "VALID_DEVICES",
    "VALID_PRECISIONS",
    "normalize_device",
    "normalize_path",
    "normalize_precision",
    "validate_training_config",
    "validated_config_dict",
]
