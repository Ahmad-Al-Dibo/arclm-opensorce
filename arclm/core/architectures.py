"""Architecture builders owned by ArcLM Core."""

from __future__ import annotations

from typing import Any, Protocol


class Architecture(Protocol):
    """Architecture contract for building runtime model instances."""

    architecture_id: str

    def build(self, config: Any, runtime: Any) -> Any: ...


class ArcLMNativeArchitecture:
    """Builder for the compact native ArcLM causal LM."""

    architecture_id = "arclm-native-causal-lm"

    def build(self, config: Any, runtime: Any) -> Any:
        from ..model import ArcLM

        return ArcLM(
            vocab_size=int(config.vocab_size),
            embed_dim=int(config.embed_dim),
            block_size=int(config.block_size),
            num_blocks=int(config.num_blocks),
            dropout=float(config.dropout),
        ).to(runtime.torch_device())
