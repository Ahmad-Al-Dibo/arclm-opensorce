"""Lightweight event and callback system for ArcLM workflows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ._version import __version__


@dataclass
class Event:
    """Typed event payload without raw dataset content."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    step: Optional[int] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1.0"
    arclm_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventHandler:
    """Base callback with method dispatch based on event type."""

    fail_fast: bool = False

    def handle(self, event: Event) -> None:
        method_name = f"on_{event.type}"
        method = getattr(self, method_name, None)
        if method is not None:
            method(event)


class CallbackManager:
    """Dispatch events to handlers with predictable failure behavior."""

    def __init__(self, handlers: Optional[Iterable[EventHandler]] = None):
        self.handlers = list(handlers or [])
        self.warnings: list[str] = []

    def add(self, handler: EventHandler) -> None:
        self.handlers.append(handler)

    def emit(self, event: Event) -> None:
        for handler in self.handlers:
            try:
                handler.handle(event)
            except Exception as exc:
                message = f"Callback {type(handler).__name__} failed for {event.type}: {exc}"
                self.warnings.append(message)
                if getattr(handler, "fail_fast", False):
                    raise


class JSONLMetricsLogger(EventHandler):
    """Write step/evaluation/checkpoint events to JSONL."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def handle(self, event: Event) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


class ConsoleProgressCallback(EventHandler):
    """Minimal console progress callback."""

    def on_step_completed(self, event: Event) -> None:
        print(f"step={event.step} metrics={event.payload}")


class EarlyStoppingCallback(EventHandler):
    """Track metric improvements and expose a stopped flag."""

    def __init__(self, metric: str = "loss", patience: int = 3, mode: str = "min"):
        self.metric = metric
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.bad_steps = 0
        self.stopped = False

    def on_evaluation_completed(self, event: Event) -> None:
        value = event.payload.get(self.metric)
        if value is None:
            return
        value = float(value)
        improved = self.best is None or (value < self.best if self.mode == "min" else value > self.best)
        if improved:
            self.best = value
            self.bad_steps = 0
        else:
            self.bad_steps += 1
            self.stopped = self.bad_steps >= self.patience


class CheckpointTrackingCallback(EventHandler):
    """Track checkpoint paths from events."""

    def __init__(self):
        self.checkpoints: list[str] = []

    def on_checkpoint_saved(self, event: Event) -> None:
        path = event.payload.get("path")
        if path:
            self.checkpoints.append(str(path))


__all__ = [
    "CallbackManager",
    "CheckpointTrackingCallback",
    "ConsoleProgressCallback",
    "EarlyStoppingCallback",
    "Event",
    "EventHandler",
    "JSONLMetricsLogger",
]
