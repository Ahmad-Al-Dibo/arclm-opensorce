"""Validation helpers for ArcLM-owned training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    """Validation metrics for one pass over a dataset."""

    loss: float | None
    batches: int
    samples: int
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {"loss": self.loss, "batches": self.batches, "samples": self.samples, "tokens": self.tokens}


class Evaluator:
    """Runs validation without owning optimizer state."""

    def evaluate(self, *, model: Any, dataloader: Any, strategy: Any, engine: Any) -> EvaluationResult:
        if dataloader is None:
            return EvaluationResult(loss=None, batches=0, samples=0, tokens=0)

        import torch

        was_training = bool(getattr(model, "training", False))
        model.eval()
        losses: list[float] = []
        samples = 0
        tokens = 0
        with torch.no_grad():
            for batch in dataloader:
                loss = strategy.loss(model=model, batch=batch, engine=engine)
                inputs, _ = engine.unpack_next_token_batch(batch)
                losses.append(engine.backend.scalar(loss))
                batch_samples, batch_tokens = engine.count_batch(inputs)
                samples += batch_samples
                tokens += batch_tokens
        if was_training:
            model.train()
        value = sum(losses) / len(losses) if losses else None
        return EvaluationResult(loss=value, batches=len(losses), samples=samples, tokens=tokens)


__all__ = ["EvaluationResult", "Evaluator"]
