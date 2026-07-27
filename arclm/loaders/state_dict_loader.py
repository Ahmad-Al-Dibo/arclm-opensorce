"""Loader for generic PyTorch state-dict files."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base import (
    CheckpointLoader,
    LoadedCheckpoint,
    load_torch_checkpoint,
    looks_like_state_dict,
)


class StateDictLoader(CheckpointLoader):
    """Load raw ``.pth/.pt/.bin/.ckpt`` files that contain model weights."""

    source_type = "state_dict"
    suffixes = {".pth", ".pt", ".bin", ".ckpt"}

    def can_load(self, source: Any) -> bool:
        path = Path(str(source))
        if not path.is_file() or path.suffix.lower() not in self.suffixes:
            return False
        try:
            checkpoint = load_torch_checkpoint(path, map_location="cpu")
        except Exception:
            return False
        return self._extract_state_dict(checkpoint) is not None

    def load(self, source: Any, map_location: str = "cpu") -> LoadedCheckpoint:
        path = Path(str(source))
        checkpoint = load_torch_checkpoint(path, map_location=map_location)
        state_dict = self._extract_state_dict(checkpoint)
        if state_dict is None:
            raise ValueError(f"No tensor state_dict found in {path}")

        config = dict(checkpoint.get("config") or {}) if isinstance(checkpoint, dict) else {}
        vocab_size = self._infer_vocab_size(state_dict, config)
        inferred = self._infer_arclm_config(state_dict)
        inferred.update(config)

        return LoadedCheckpoint(
            source=str(path),
            source_type=self.source_type,
            state_dict=dict(state_dict),
            config=inferred,
            vocab_size=vocab_size,
            metadata={"path": str(path), "format": path.suffix.lower()},
        )

    def _extract_state_dict(self, checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
        if looks_like_state_dict(checkpoint):
            return dict(checkpoint)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if looks_like_state_dict(value):
                    return dict(value)
        return None

    def _infer_vocab_size(self, state_dict, config):
        if config.get("vocab_size") is not None:
            return int(config["vocab_size"])
        for key in ("token_embedding.weight", "embedding.weight", "head.weight"):
            tensor = state_dict.get(key)
            if torch.is_tensor(tensor) and tensor.ndim >= 2:
                return int(tensor.shape[0])
        return None

    def _infer_arclm_config(self, state_dict):
        config = {}
        token_embedding = state_dict.get("token_embedding.weight")
        position_embedding = state_dict.get("position_embedding.weight")
        if torch.is_tensor(token_embedding) and token_embedding.ndim == 2:
            config["vocab_size"] = int(token_embedding.shape[0])
            config["embed_dim"] = int(token_embedding.shape[1])
        if torch.is_tensor(position_embedding) and position_embedding.ndim == 2:
            config["block_size"] = int(position_embedding.shape[0])
            config.setdefault("embed_dim", int(position_embedding.shape[1]))

        block_indices = set()
        for name in state_dict:
            parts = name.split(".")
            if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit():
                block_indices.add(int(parts[1]))
        if block_indices:
            config["num_blocks"] = max(block_indices) + 1
        return config
