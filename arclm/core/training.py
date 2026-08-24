"""ArcLM-owned training engine and strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

from .backend import TorchBackend


class TrainingStrategy(Protocol):
    """Mode-specific behavior used by the shared training engine."""

    name: str

    def prepare(self, *, model: Any, engine: "TrainingEngine") -> None: ...
    def loss(self, *, model: Any, batch: Any, engine: "TrainingEngine") -> Any: ...


@dataclass
class TrainingEngineConfig:
    """Configuration owned by the training engine."""

    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = None
    checkpoint_interval: int | None = None
    validate_interval: int | None = None


@dataclass
class TrainingResult:
    """Structured result returned by the engine."""

    train_losses: list[float] = field(default_factory=list)
    step_losses: list[float] = field(default_factory=list)
    optimizer_steps: int = 0
    scheduler_steps: int = 0
    global_steps: int = 0
    strategy: str = "pretrain"
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_history(self) -> dict[str, Any]:
        """Return legacy-compatible history keys plus engine metrics."""

        return {
            "train_losses": list(self.train_losses),
            "step_losses": list(self.step_losses),
            "optimizer_steps": self.optimizer_steps,
            "scheduler_steps": self.scheduler_steps,
            "global_steps": self.global_steps,
            "strategy": self.strategy,
            "metrics": dict(self.metrics),
        }


class PretrainStrategy:
    """Next-token pretraining strategy."""

    name = "pretrain"

    def prepare(self, *, model: Any, engine: "TrainingEngine") -> None:
        model.train()

    def loss(self, *, model: Any, batch: Any, engine: "TrainingEngine") -> Any:
        inputs, targets = engine.unpack_next_token_batch(batch)
        logits = model(inputs)
        return engine.backend.cross_entropy_next_token_loss(logits, targets)


class FullFineTuneStrategy(PretrainStrategy):
    """Full fine-tuning uses the same current next-token loop with all params trainable."""

    name = "full_finetune"

    def prepare(self, *, model: Any, engine: "TrainingEngine") -> None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        model.train()


class AdapterStrategy(PretrainStrategy):
    """Adapter fine-tuning freezes base params and trains adapter params."""

    name = "adapter"

    def __init__(self, adapter_type: str = "lora"):
        self.adapter_type = adapter_type

    def prepare(self, *, model: Any, engine: "TrainingEngine") -> None:
        trainable = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora_" in name
            if parameter.requires_grad:
                trainable += parameter.numel()
        if trainable == 0:
            raise ValueError("AdapterStrategy requires attached adapter parameters.")
        model.train()


CheckpointHook = Callable[["TrainingEngine", dict[str, Any]], None]
ValidationHook = Callable[["TrainingEngine", Any], dict[str, Any] | None]


class TrainingEngine:
    """One ArcLM training engine shared by mode-specific strategies."""

    def __init__(self, *, backend: Any | None = None, config: TrainingEngineConfig | None = None, runtime: Any | None = None):
        self.backend = backend or TorchBackend()
        self.config = config or TrainingEngineConfig()
        self.runtime = runtime
        self.global_step = 0
        self.result = TrainingResult()

    def fit(
        self,
        *,
        model: Any,
        dataloader: Any,
        strategy: TrainingStrategy,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        checkpoint_hook: CheckpointHook | None = None,
        validation_hook: ValidationHook | None = None,
        validation_data: Any | None = None,
        debug: bool = False,
    ) -> TrainingResult:
        """Run the training lifecycle."""

        strategy.prepare(model=model, engine=self)
        optimizer = optimizer or self.backend.create_optimizer(
            model.parameters(),
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        result = TrainingResult(strategy=strategy.name)
        self.result = result

        for epoch in range(self.config.epochs):
            epoch_losses: list[float] = []
            for batch in dataloader:
                self.backend.zero_grad(optimizer)
                loss = strategy.loss(model=model, batch=batch, engine=self)
                self.backend.backward(loss)
                self.backend.clip_grad_norm(model.parameters(), self.config.grad_clip)
                self.backend.optimizer_step(optimizer)
                self.global_step += 1
                result.global_steps = self.global_step
                result.optimizer_steps += 1
                loss_value = self.backend.scalar(loss)
                result.step_losses.append(loss_value)
                epoch_losses.append(loss_value)

                # view checkpoint hook after each step if due and printing the loss for debugging
                if debug:
                    print(f"Epoch {epoch + 1}/{self.config.epochs}, Step {self.global_step}, Loss: {loss_value:.4f}")

                if checkpoint_hook is not None and self._due(self.config.checkpoint_interval):
                    checkpoint_hook(self, {"event": "step", "epoch": epoch + 1, "step": self.global_step})

            if scheduler is not None:
                metric = sum(epoch_losses) / len(epoch_losses) if epoch_losses else None
                self.backend.scheduler_step(scheduler, metric)
                result.scheduler_steps += 1

            if epoch_losses:
                result.train_losses.append(sum(epoch_losses) / len(epoch_losses))

            if validation_hook is not None and self.config.validate_interval and (epoch + 1) % self.config.validate_interval == 0:
                metrics = validation_hook(self, validation_data)
                if metrics:
                    result.metrics[f"validation_epoch_{epoch + 1}"] = metrics

            if checkpoint_hook is not None:
                checkpoint_hook(self, {"event": "epoch", "epoch": epoch + 1, "step": self.global_step})

        model.eval()
        return result

    def unpack_next_token_batch(self, batch: Any) -> tuple[Any, Any]:
        """Normalize supported batch shapes for next-token training."""

        inputs: Any
        targets: Any
        if isinstance(batch, dict):
            inputs = batch.get("input_ids") or batch.get("inputs") or batch.get("x")
            targets = batch.get("labels") or batch.get("targets") or batch.get("y")
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0], batch[1]
        else:
            raise ValueError("Training batch must contain inputs and targets.")

        if inputs is None or targets is None:
            raise ValueError("Training batch must contain inputs and targets.")
        if self.runtime is not None:
            device = self.runtime.torch_device()
            inputs = inputs.to(device)
            targets = targets.to(device)
        return inputs, targets

    def trainable_parameters(self, model: Any) -> Iterable[Any]:
        """Return trainable parameters for inspection/tests."""

        return (parameter for parameter in model.parameters() if getattr(parameter, "requires_grad", False))

    def _due(self, interval: int | None) -> bool:
        return bool(interval and self.global_step > 0 and self.global_step % interval == 0)
