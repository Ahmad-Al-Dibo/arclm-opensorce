"""Structured training metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StepMetrics:
    """Metrics produced after one training step."""

    epoch: int
    step: int
    optimizer_step: int
    train_loss: float
    epoch_step: int = 0
    steps_per_epoch: int | None = None
    validation_loss: float | None = None
    learning_rate: float | None = None
    gradient_norm: float | None = None
    tokens_processed: int = 0
    samples_processed: int = 0
    elapsed_time: float = 0.0
    checkpoint: str | None = None
    progress: float = 0.0
    stopping_status: str = "running"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingResult:
    """Structured result returned by the engine."""

    strategy: str
    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    steps: list[StepMetrics] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    optimizer_steps: int = 0
    scheduler_steps: int = 0
    global_steps: int = 0
    tokens_processed: int = 0
    samples_processed: int = 0
    elapsed_time: float = 0.0
    stopping_status: str = "completed"
    stopping_reason: str = "completed"
    resumed_from: str | None = None
    artifact: str | None = None
    adapter: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_history(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "train_losses": list(self.train_losses),
            "validation_losses": list(self.validation_losses),
            "step_losses": [step.train_loss for step in self.steps],
            "steps": [step.to_dict() for step in self.steps],
            "checkpoints": list(self.checkpoints),
            "optimizer_steps": self.optimizer_steps,
            "scheduler_steps": self.scheduler_steps,
            "global_steps": self.global_steps,
            "tokens_processed": self.tokens_processed,
            "samples_processed": self.samples_processed,
            "elapsed_time": self.elapsed_time,
            "stopping_status": self.stopping_status,
            "stopping_reason": self.stopping_reason,
            "resumed_from": self.resumed_from,
            "artifact": self.artifact,
            "adapter": self.adapter,
            "metrics": dict(self.metrics),
        }


__all__ = ["StepMetrics", "TrainingResult"]
