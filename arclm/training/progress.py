"""Console progress display for training."""

from __future__ import annotations

import sys
from typing import Any

from .events import TrainingEvent


class ConsoleProgress:
    """Small single-line training progress display."""

    def __init__(self, stream: Any | None = None):
        self.stream = stream or sys.stderr
        self._active = False

    def on_train_start(self, event: TrainingEvent) -> None:
        self._active = True
        self._write("Starting training...")

    def on_step_end(self, event: TrainingEvent) -> None:
        metrics = event.metrics
        epoch = metrics.get("epoch", event.epoch)
        epoch_step = metrics.get("epoch_step", event.step)
        steps_per_epoch = metrics.get("steps_per_epoch") or "?"
        loss = metrics.get("train_loss")
        progress = float(metrics.get("progress", 0.0)) * 100.0
        status = metrics.get("stopping_status", "running")
        loss_text = f"{float(loss):.4f}" if loss is not None else "n/a"
        self._write(
            f"Epoch {epoch} | Step {epoch_step}/{steps_per_epoch} | "
            f"Loss {loss_text} | Progress {progress:5.1f}% | Status {status}"
        )

    def on_validation_end(self, event: TrainingEvent) -> None:
        loss = event.metrics.get("loss")
        loss_text = f"{float(loss):.4f}" if loss is not None else "n/a"
        self._write(f"Epoch {event.epoch} | Validation loss {loss_text} | Status validating")

    def on_train_end(self, event: TrainingEvent) -> None:
        status = event.metrics.get("stopping_status", "completed")
        reason = event.metrics.get("stopping_reason", "completed")
        self._write(f"Training {status} | Reason {reason}")
        self.stream.write("\n")
        self.stream.flush()
        self._active = False

    def on_error(self, event: TrainingEvent) -> None:
        self._write("Training stopped | Reason error")
        self.stream.write("\n")
        self.stream.flush()
        self._active = False

    def _write(self, message: str) -> None:
        self.stream.write("\r" + message)
        self.stream.flush()


__all__ = ["ConsoleProgress"]
