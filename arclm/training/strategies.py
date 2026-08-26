"""Training and fine-tuning strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import FineTuningConfig
from .losses import LossFunction, NextTokenLoss


class TrainingStrategy(Protocol):
    """Mode-specific behavior used by the shared training engine."""

    name: str

    def prepare(self, *, model: Any, engine: Any) -> None: ...
    def loss(self, *, model: Any, batch: Any, engine: Any) -> Any: ...
    def inspect_trainable(self, model: Any) -> dict[str, Any]: ...


@dataclass
class BaseStrategy:
    """Shared next-token language-model strategy behavior."""

    name: str = "pretrain"
    loss_function: LossFunction = field(default_factory=NextTokenLoss)

    def prepare(self, *, model: Any, engine: Any) -> None:
        model.train()

    def loss(self, *, model: Any, batch: Any, engine: Any) -> Any:
        return self.loss_function(model=model, batch=batch, engine=engine)

    def inspect_trainable(self, model: Any) -> dict[str, Any]:
        total = 0
        trainable = 0
        trainable_names: list[str] = []
        frozen_names: list[str] = []
        for name, parameter in model.named_parameters():
            count = int(parameter.numel())
            total += count
            if getattr(parameter, "requires_grad", False):
                trainable += count
                trainable_names.append(name)
            else:
                frozen_names.append(name)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_names": trainable_names,
            "frozen_names": frozen_names,
        }


class PretrainStrategy(BaseStrategy):
    """Train all model parameters for next-token modeling."""

    def __init__(self):
        super().__init__(name="pretrain")

    def prepare(self, *, model: Any, engine: Any) -> None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()


class FullFineTuneStrategy(BaseStrategy):
    """Fine-tune either all parameters or an explicit parameter subset."""

    def __init__(self, config: FineTuningConfig | None = None):
        super().__init__(name="full_finetune")
        self.config = config or FineTuningConfig(method="full")

    def prepare(self, *, model: Any, engine: Any) -> None:
        configure_trainable_parameters(
            model,
            freeze_base_model=self.config.freeze_base_model,
            trainable_patterns=self.config.trainable_patterns,
            freeze_patterns=self.config.freeze_patterns,
        )
        model.train()


class AdapterStrategy(BaseStrategy):
    """Parameter-efficient fine-tuning strategy for attached adapters."""

    def __init__(self, config: FineTuningConfig | None = None):
        super().__init__(name="adapter")
        self.config = config or FineTuningConfig(method="lora", freeze_base_model=True)

    def prepare(self, *, model: Any, engine: Any) -> None:
        configure_trainable_parameters(
            model,
            freeze_base_model=True,
            trainable_patterns=self.config.trainable_patterns or ("lora_",),
            freeze_patterns=self.config.freeze_patterns,
        )
        report = self.inspect_trainable(model)
        if report["trainable_parameters"] == 0:
            raise ValueError("Adapter fine-tuning requires attached adapter parameters.")
        model.train()


def configure_trainable_parameters(
    model: Any,
    *,
    freeze_base_model: bool = False,
    trainable_patterns: tuple[str, ...] = (),
    freeze_patterns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Apply ArcLM-owned freezing and trainable parameter selection."""

    for name, parameter in model.named_parameters():
        trainable = not freeze_base_model
        if trainable_patterns:
            trainable = any(pattern in name for pattern in trainable_patterns)
        if freeze_patterns and any(pattern in name for pattern in freeze_patterns):
            trainable = False
        parameter.requires_grad = trainable
    return BaseStrategy().inspect_trainable(model)


__all__ = [
    "AdapterStrategy",
    "BaseStrategy",
    "FullFineTuneStrategy",
    "PretrainStrategy",
    "TrainingStrategy",
    "configure_trainable_parameters",
]
