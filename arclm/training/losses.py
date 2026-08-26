"""Loss contracts used by ArcLM training strategies."""

from __future__ import annotations

from typing import Any, Protocol


class LossFunction(Protocol):
    """Callable loss component."""

    def __call__(self, *, model: Any, batch: Any, engine: Any) -> Any: ...


class NextTokenLoss:
    """Causal next-token language-model loss."""

    def __call__(self, *, model: Any, batch: Any, engine: Any) -> Any:
        inputs, targets = engine.unpack_next_token_batch(batch)
        logits = model(inputs)
        return engine.backend.cross_entropy_next_token_loss(logits, targets)


__all__ = ["LossFunction", "NextTokenLoss"]
