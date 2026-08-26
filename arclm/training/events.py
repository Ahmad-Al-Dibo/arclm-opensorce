"""Training events and callback dispatch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TrainingEvent:
    """Structured event emitted by the training engine."""

    name: str
    epoch: int = 0
    step: int = 0
    optimizer_step: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    elapsed_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingCallback(Protocol):
    """Optional lifecycle hooks for progress UIs, logs, and integrations."""

    def on_train_start(self, event: TrainingEvent) -> None: ...
    def on_step_end(self, event: TrainingEvent) -> None: ...
    def on_validation_end(self, event: TrainingEvent) -> None: ...
    def on_checkpoint(self, event: TrainingEvent) -> None: ...
    def on_train_end(self, event: TrainingEvent) -> None: ...
    def on_error(self, event: TrainingEvent) -> None: ...


class CallbackManager:
    """Dispatch events to partially implemented callback objects."""

    def __init__(self, callbacks: list[Any] | tuple[Any, ...] | None = None):
        self.callbacks = list(callbacks or [])
        self.events: list[TrainingEvent] = []

    def emit(self, event: TrainingEvent) -> None:
        self.events.append(event)
        handler_name = f"on_{event.name}"
        for callback in self.callbacks:
            handler = getattr(callback, handler_name, None)
            if handler is not None:
                handler(event)


__all__ = ["CallbackManager", "TrainingCallback", "TrainingEvent"]
