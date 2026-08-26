"""Training configuration and inspectable plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FineTuningConfig:
    """Fine-tuning behavior owned by ArcLM."""

    method: str = "full"
    freeze_base_model: bool = False
    trainable_patterns: tuple[str, ...] = ()
    freeze_patterns: tuple[str, ...] = ()
    adapter_type: str = "lora"
    rank: int = 4
    alpha: float = 8.0
    target_modules: tuple[str, ...] = ()

    def normalized_method(self) -> str:
        method = str(self.method or "full").lower().strip().replace("-", "_")
        if method in {"full", "full_finetune", "finetune"}:
            return "full_finetune"
        if method in {"adapter", "lora", "peft"}:
            return "adapter"
        if method == "pretrain":
            return "pretrain"
        raise ValueError("fine-tuning method must be one of: full, full_finetune, adapter, lora, peft.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        payload["trainable_patterns"] = list(self.trainable_patterns)
        payload["freeze_patterns"] = list(self.freeze_patterns)
        return payload


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration consumed by the ArcLM training engine."""

    epochs: int = 1
    batch_size: int = 2
    block_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = None
    gradient_accumulation_steps: int = 1
    steps_per_epoch: int | None = None
    checkpoint_interval: int | None = None
    validation_interval: int | None = 1
    log_interval: int | None = None
    max_steps: int | None = None
    early_stopping: bool = False
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    early_stopping_metric: str = "validation_loss"
    show_progress: bool = False
    shuffle: bool = True
    checkpoint_dir: Path | None = None
    resume_from: Path | None = None
    save_artifact: bool = False
    artifact_path: Path | None = None
    save_adapter: bool = False
    adapter_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.steps_per_epoch is not None and self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive when provided.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided.")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative when provided.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("checkpoint_dir", "resume_from", "artifact_path", "adapter_path"):
            value = payload.get(key)
            payload[key] = str(value) if value is not None else None
        return payload


@dataclass(frozen=True)
class TrainingPlan:
    """Inspectable public training plan."""

    strategy: str
    epochs: int
    batch_size: int
    learning_rate: float
    block_size: int
    runtime: dict[str, Any]
    gradient_accumulation_steps: int = 1
    steps_per_epoch: int | None = None
    checkpoint_interval: int | None = None
    validation_interval: int | None = None
    max_steps: int | None = None
    early_stopping: dict[str, Any] | None = None
    fine_tuning: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        return self.to_dict()


__all__ = ["FineTuningConfig", "TrainingConfig", "TrainingPlan"]
