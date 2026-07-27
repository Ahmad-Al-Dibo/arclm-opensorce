"""Adapt normalized checkpoints into ArcLM training bundles."""

from typing import Any, Dict, Optional

import torch

from ..config import Config
from ..model import ArcLM
from .base import AdaptedModelBundle, LoadedCheckpoint


_CONFIG_KEYS = {
    "embed_dim",
    "block_size",
    "batch_size",
    "num_epochs",
    "vocab_size",
    "learning_rate",
    "weight_decay",
    "dropout",
    "grad_clip",
    "num_blocks",
    "model_path",
    "tokenizer_path",
    "data_path",
    "domain_data_path",
    "domain_data_repeats",
    "tokenizer_type",
    "sentencepiece_model_type",
    "sentencepiece_character_coverage",
    "tokenizer_max_line_length",
    "max_vocab",
    "max_data_size",
    "device",
    "validation_split",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "restore_best_model",
    "seed",
    "diagnostic_top_k",
    "concept_benchmark_top_k",
    "diagnostic_prompts",
    "diagnostic_sample_tokens",
    "tokenizer_rare_threshold",
    "training_log_interval",
    "run_long_context_evaluation",
    "use_checkpoint_tokenizer",
    "long_context_block_sizes",
    "freeze_backbone",
    "freeze_embedding",
    "use_discriminative_lr",
    "lr_multiplier",
    "use_lr_scheduler",
    "lr_scheduler_strategy",
    "warmup_epochs",
    "checkpoint_interval",
    "checkpoint_batch_interval",
    "metrics_log_path",
}


def config_from_checkpoint(
    checkpoint: LoadedCheckpoint,
    target_config: Optional[Config] = None,
    **overrides,
) -> Config:
    """Create a Config using ArcLM-compatible checkpoint fields plus overrides."""

    if target_config is not None:
        data = target_config.to_dict()
        for key in (
            "freeze_backbone",
            "freeze_embedding",
            "use_discriminative_lr",
            "lr_multiplier",
            "use_lr_scheduler",
            "lr_scheduler_strategy",
            "warmup_epochs",
            "checkpoint_interval",
            "checkpoint_batch_interval",
            "metrics_log_path",
        ):
            if hasattr(target_config, key):
                data[key] = getattr(target_config, key)
    else:
        data = {}

    for key, value in checkpoint.config.items():
        if key in _CONFIG_KEYS and value is not None:
            data.setdefault(key, value)

    if checkpoint.vocab_size is not None:
        data.setdefault("vocab_size", checkpoint.vocab_size)
        data.setdefault("max_vocab", checkpoint.vocab_size)

    data.update({key: value for key, value in overrides.items() if value is not None})
    if "vocab_size" not in data or data["vocab_size"] is None:
        raise ValueError("A vocab_size is required to adapt a checkpoint for ArcLM.")

    config = Config(**data)
    return config


def validate_tokenizer_compatibility(
    checkpoint: LoadedCheckpoint,
    tokenizer: Optional[Any] = None,
    config: Optional[Config] = None,
) -> None:
    """Raise if a tokenizer cannot safely address the checkpoint vocabulary."""

    expected_vocab_size = checkpoint.vocab_size
    if config is not None and config.vocab_size is not None:
        expected_vocab_size = config.vocab_size

    if tokenizer is not None:
        actual_vocab_size = tokenizer.get_vocab_size()
        if expected_vocab_size is not None and actual_vocab_size != expected_vocab_size:
            raise ValueError(
                "Tokenizer vocabulary size is incompatible with checkpoint: "
                f"checkpoint={expected_vocab_size}, tokenizer={actual_vocab_size}."
            )

        checkpoint_stoi = checkpoint.stoi
        tokenizer_stoi = getattr(tokenizer, "stoi", None)
        if checkpoint_stoi is not None and tokenizer_stoi is not None:
            normalized = {str(token): int(index) for token, index in tokenizer_stoi.items()}
            if checkpoint_stoi != normalized:
                raise ValueError(
                    "Tokenizer token-to-id mapping differs from the checkpoint. "
                    "Reuse the checkpoint tokenizer for fine-tuning or train a new head."
                )
        return

    if checkpoint.source_type in {"arclm", "huggingface"} and checkpoint.stoi is None:
        metadata_type = checkpoint.tokenizer_metadata.get("tokenizer_type")
        if metadata_type not in {None, "huggingface"}:
            raise ValueError(
                "Checkpoint declares tokenizer metadata but no tokenizer object was provided."
            )


def adapt_for_training(
    checkpoint: LoadedCheckpoint,
    target_config: Optional[Config] = None,
    tokenizer: Optional[Any] = None,
    strict: bool = False,
    require_tokenizer_match: bool = False,
    **config_overrides,
) -> AdaptedModelBundle:
    """Create an ArcLM model from a normalized checkpoint."""

    config = config_from_checkpoint(checkpoint, target_config, **config_overrides)
    if require_tokenizer_match:
        validate_tokenizer_compatibility(checkpoint, tokenizer=tokenizer, config=config)

    model = ArcLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        block_size=config.block_size,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
    ).to(torch.device(config.device))

    state_dict = checkpoint.require_state_dict()
    copied_state = _copy_matching_weights(model.state_dict(), state_dict)
    load_result = model.load_state_dict(copied_state, strict=False)

    if strict and (load_result.missing_keys or load_result.unexpected_keys):
        missing = ", ".join(load_result.missing_keys)
        unexpected = ", ".join(load_result.unexpected_keys)
        raise ValueError(f"Strict adaptation failed. Missing: {missing}. Unexpected: {unexpected}.")

    return AdaptedModelBundle(
        model=model,
        config=config,
        checkpoint=checkpoint,
        missing_keys=list(load_result.missing_keys),
        unexpected_keys=list(load_result.unexpected_keys),
        tokenizer_metadata=dict(checkpoint.tokenizer_metadata or {}),
    )


def _copy_matching_weights(target_state: Dict[str, torch.Tensor], source_state: Dict[str, torch.Tensor]):
    copied = {}
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if torch.is_tensor(source_tensor) and tuple(source_tensor.shape) == tuple(target_tensor.shape):
            copied[name] = source_tensor.to(dtype=target_tensor.dtype)
        else:
            copied[name] = target_tensor
    return copied
