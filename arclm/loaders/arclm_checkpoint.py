"""Loader for native ArcLM checkpoints."""

from pathlib import Path
from typing import Any

from .base import (
    CheckpointLoader,
    LoadedCheckpoint,
    load_torch_checkpoint,
    normalize_itos,
    normalize_stoi,
)


class ArcLMCheckpointLoader(CheckpointLoader):
    """Load checkpoints produced by ``Trainer.save``."""

    source_type = "arclm"

    def can_load(self, source: Any) -> bool:
        path = Path(str(source))
        if not path.is_file() or path.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
            return False
        try:
            checkpoint = load_torch_checkpoint(path, map_location="cpu")
        except Exception:
            return False
        return isinstance(checkpoint, dict) and any(
            key in checkpoint for key in ("model_state_dict", "model")
        ) and (
            "config" in checkpoint
            or "stoi" in checkpoint
            or "tokenizer_metadata" in checkpoint
        )

    def load(self, source: Any, map_location: str = "cpu") -> LoadedCheckpoint:
        path = Path(str(source))
        checkpoint = load_torch_checkpoint(path, map_location=map_location)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"ArcLM checkpoint must be a dictionary: {path}")

        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model")
        if state_dict is None:
            raise ValueError(f"ArcLM checkpoint has no model weights: {path}")

        config = dict(checkpoint.get("config") or {})
        vocab_size = checkpoint.get("vocab_size") or config.get("vocab_size")

        return LoadedCheckpoint(
            source=str(path),
            source_type=self.source_type,
            state_dict=dict(state_dict),
            config=config,
            vocab_size=int(vocab_size) if vocab_size is not None else None,
            tokenizer_metadata=dict(checkpoint.get("tokenizer_metadata") or {}),
            vocab=checkpoint.get("vocab"),
            stoi=normalize_stoi(checkpoint.get("stoi")),
            itos=normalize_itos(checkpoint.get("itos")),
            optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
            train_history=dict(checkpoint.get("train_history") or {}),
            metadata={
                "path": str(path),
                "current_epoch": checkpoint.get("current_epoch", 0),
                "best_val_loss": checkpoint.get("best_val_loss"),
            },
        )
