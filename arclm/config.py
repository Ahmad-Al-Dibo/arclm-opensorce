"""Configuration objects for the current ArcLM architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Config:
    """Native model and training configuration."""

    vocab_size: int | None = None
    embed_dim: int = 64
    block_size: int = 8
    num_blocks: int = 2
    dropout: float = 0.0
    batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = None
    max_vocab: int = 50000
    device: str = "cpu"
    architecture: str | None = None
    model_family: str | None = None
    external_model_id: str | None = None
    compatibility_backend: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    torch_dtype: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.vocab_size is not None and int(self.vocab_size) <= 0:
            raise ValueError("vocab_size must be positive when provided.")
        if int(self.embed_dim) <= 0:
            raise ValueError("embed_dim must be positive.")
        if int(self.block_size) <= 0:
            raise ValueError("block_size must be positive.")
        if int(self.num_blocks) <= 0:
            raise ValueError("num_blocks must be positive.")
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(self.num_epochs) <= 0:
            raise ValueError("num_epochs must be positive.")
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["Config"]
