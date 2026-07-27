"""Loader for safetensors checkpoint files."""

from pathlib import Path
from typing import Any

import torch

from .base import CheckpointLoader, LoadedCheckpoint
from .state_dict_loader import StateDictLoader


class SafetensorsLoader(CheckpointLoader):
    """Load ``.safetensors`` files when the optional package is installed."""

    source_type = "safetensors"

    def can_load(self, source: Any) -> bool:
        return Path(str(source)).is_file() and Path(str(source)).suffix.lower() == ".safetensors"

    def load(self, source: Any, map_location: str = "cpu") -> LoadedCheckpoint:
        path = Path(str(source))
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors files requires the safetensors package. "
                "Install it with: pip install safetensors"
            ) from exc

        state_dict = load_file(str(path), device=map_location)
        helper = StateDictLoader()
        config = helper._infer_arclm_config(state_dict)
        vocab_size = helper._infer_vocab_size(state_dict, config)

        return LoadedCheckpoint(
            source=str(path),
            source_type=self.source_type,
            state_dict={name: tensor for name, tensor in state_dict.items() if torch.is_tensor(tensor)},
            config=config,
            vocab_size=vocab_size,
            metadata={"path": str(path), "format": ".safetensors"},
        )
