"""ArcLM-owned training engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..core.backend import TorchBackend
from .checkpoints import CheckpointManager
from .config import TrainingConfig
from .events import CallbackManager, TrainingEvent
from .evaluation import Evaluator
from .metrics import StepMetrics, TrainingResult
from .progress import ConsoleProgress
from .strategies import TrainingStrategy


class TrainingCancelled(RuntimeError):
    """Raised by callbacks or integrations to stop training early."""


@dataclass
class ResumeState:
    """Restored training state."""

    epoch: int = 0
    step: int = 0
    optimizer_step: int = 0
    checkpoint: str | None = None


class TrainingEngine:
    """Coordinates ArcLM's training lifecycle."""

    def __init__(
        self,
        *,
        backend: Any | None = None,
        config: TrainingConfig | None = None,
        runtime: Any | None = None,
        evaluator: Evaluator | None = None,
        callbacks: list[Any] | tuple[Any, ...] | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ):
        self.backend = backend or TorchBackend()
        self.config = config or TrainingConfig()
        self.runtime = runtime
        self.evaluator = evaluator or Evaluator()
        self.callbacks = CallbackManager(callbacks)
        self.checkpoints = checkpoint_manager or (CheckpointManager(self.config.checkpoint_dir) if self.config.checkpoint_dir else None)
        self.global_step = 0
        self.optimizer_step = 0
        self.result = TrainingResult(strategy="unknown")

    def fit(
        self,
        *,
        model: Any,
        dataloader: Any,
        strategy: TrainingStrategy,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        validation_loader: Any | None = None,
        debug: bool = False,
    ) -> TrainingResult:
        """Run the full training lifecycle."""

        started = time.monotonic()
        result = TrainingResult(strategy=strategy.name)
        self.result = result
        if debug or self.config.show_progress:
            self.callbacks.callbacks.append(ConsoleProgress())
        strategy.prepare(model=model, engine=self)
        optimizer = optimizer or self.backend.create_optimizer(
            model.parameters(),
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        resume_state = self._resume_if_requested(model=model, optimizer=optimizer, scheduler=scheduler)
        if resume_state.checkpoint:
            result.resumed_from = resume_state.checkpoint
            self.global_step = resume_state.step
            self.optimizer_step = resume_state.optimizer_step

        trainable_report = strategy.inspect_trainable(model)
        result.metrics["trainable"] = trainable_report
        planned_steps = self._planned_total_steps(dataloader)
        self._emit("train_start", elapsed=time.monotonic() - started, metrics={"trainable": trainable_report, "planned_steps": planned_steps})

        try:
            current_epoch = max(1, resume_state.epoch + 1)
            best_metric: float | None = None
            unimproved_epochs = 0
            stop_training = False
            stop_reason = "completed"
            for epoch in range(current_epoch, self.config.epochs + 1):
                current_epoch = epoch
                epoch_losses: list[float] = []
                self.backend.zero_grad(optimizer)
                epoch_step_limit = self._epoch_step_limit(dataloader)
                for epoch_step, batch in self._epoch_batches(dataloader, epoch_step_limit):
                    if self.config.max_steps is not None and self.global_step >= self.config.max_steps:
                        stop_training = True
                        stop_reason = "max_steps"
                        break

                    loss = strategy.loss(model=model, batch=batch, engine=self)
                    scaled_loss = loss / int(self.config.gradient_accumulation_steps)
                    self.backend.backward(scaled_loss)
                    self.global_step += 1
                    result.global_steps = self.global_step
                    loss_value = self.backend.scalar(loss)
                    epoch_losses.append(loss_value)

                    inputs, _ = self.unpack_next_token_batch(batch)
                    batch_samples, batch_tokens = self.count_batch(inputs)
                    result.samples_processed += batch_samples
                    result.tokens_processed += batch_tokens

                    should_step = self._should_step(epoch_step, total_batches=epoch_step_limit)
                    gradient_norm = None
                    if should_step:
                        gradient_norm = self.backend.clip_grad_norm(model.parameters(), self.config.grad_clip)
                        self.backend.optimizer_step(optimizer)
                        self.backend.zero_grad(optimizer)
                        self.optimizer_step += 1
                        result.optimizer_steps = self.optimizer_step
                        if scheduler is not None:
                            self.backend.scheduler_step(scheduler, None)
                            result.scheduler_steps += 1

                    checkpoint_path = self._save_checkpoint_if_due(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        strategy=strategy,
                        metrics={"train_loss": loss_value},
                    )
                    if checkpoint_path:
                        result.checkpoints.append(str(checkpoint_path))

                    step_metrics = StepMetrics(
                        epoch=epoch,
                        step=self.global_step,
                        optimizer_step=self.optimizer_step,
                        train_loss=loss_value,
                        epoch_step=epoch_step,
                        steps_per_epoch=epoch_step_limit,
                        learning_rate=self.current_learning_rate(optimizer),
                        gradient_norm=gradient_norm,
                        tokens_processed=result.tokens_processed,
                        samples_processed=result.samples_processed,
                        elapsed_time=time.monotonic() - started,
                        checkpoint=str(checkpoint_path) if checkpoint_path else None,
                        progress=self._progress(planned_steps),
                        stopping_status="running",
                    )
                    result.steps.append(step_metrics)
                    self._emit("step_end", epoch=epoch, metrics=step_metrics.to_dict(), checkpoint=step_metrics.checkpoint, elapsed=step_metrics.elapsed_time)

                if epoch_losses:
                    result.train_losses.append(sum(epoch_losses) / len(epoch_losses))

                validation = self._validate_if_due(model=model, validation_loader=validation_loader, strategy=strategy, epoch=epoch, elapsed=time.monotonic() - started)
                if validation is not None:
                    result.validation_losses.append(validation)
                    result.metrics[f"validation_epoch_{epoch}"] = {"loss": validation}

                if stop_training:
                    break

                if self.config.early_stopping:
                    metric = self._early_stopping_metric(validation=validation, train_losses=epoch_losses)
                    if metric is not None:
                        if best_metric is None or metric < best_metric - float(self.config.early_stopping_min_delta):
                            best_metric = metric
                            unimproved_epochs = 0
                        else:
                            unimproved_epochs += 1
                        patience = 0 if self.config.early_stopping_patience is None else int(self.config.early_stopping_patience)
                        if unimproved_epochs > patience:
                            stop_training = True
                            stop_reason = "early_stopping"
                            break

            final_checkpoint = self._save_epoch_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=current_epoch,
                strategy=strategy,
                metrics={"event": "train_end"},
            )
            if final_checkpoint:
                result.checkpoints.append(str(final_checkpoint))
            model.eval()
            result.elapsed_time = time.monotonic() - started
            result.stopping_status = "stopped" if stop_reason != "completed" else "completed"
            result.stopping_reason = stop_reason
            result.metrics["stopping"] = {
                "status": result.stopping_status,
                "reason": result.stopping_reason,
                "epoch": current_epoch,
                "step": self.global_step,
            }
            self._emit("train_end", elapsed=result.elapsed_time, metrics=result.to_history(), checkpoint=str(final_checkpoint) if final_checkpoint else None)
            return result
        except BaseException as exc:
            self._emit("error", elapsed=time.monotonic() - started, metrics={"error": repr(exc)})
            raise

    def unpack_next_token_batch(self, batch: Any) -> tuple[Any, Any]:
        """Normalize supported batch shapes for next-token training."""

        if isinstance(batch, dict):
            inputs = self._first_present(batch, ("input_ids", "inputs", "x"))
            targets = self._first_present(batch, ("labels", "targets", "y"))
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

    def count_batch(self, inputs: Any) -> tuple[int, int]:
        shape = tuple(getattr(inputs, "shape", ()))
        if not shape:
            return 1, 1
        samples = int(shape[0])
        tokens = int(samples * shape[1]) if len(shape) > 1 else samples
        return samples, tokens

    def trainable_parameters(self, model: Any) -> Iterable[Any]:
        return (parameter for parameter in model.parameters() if getattr(parameter, "requires_grad", False))

    @staticmethod
    def _first_present(batch: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in batch:
                return batch[key]
        return None

    @staticmethod
    def current_learning_rate(optimizer: Any) -> float | None:
        groups = getattr(optimizer, "param_groups", None)
        if not groups:
            return None
        return float(groups[0].get("lr", 0.0))

    def _resume_if_requested(self, *, model: Any, optimizer: Any, scheduler: Any | None) -> ResumeState:
        resume_from = self.config.resume_from
        if resume_from is None:
            return ResumeState()
        manager = self.checkpoints or CheckpointManager(Path(resume_from).parent)
        loaded = manager.load(resume_from, map_location=self.runtime.device_name if self.runtime is not None else "cpu")
        model.load_state_dict(loaded.state["model"], strict=True)
        if loaded.state.get("optimizer") is not None:
            optimizer.load_state_dict(loaded.state["optimizer"])
        if scheduler is not None and loaded.state.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(loaded.state["scheduler"])
        return ResumeState(
            epoch=loaded.metadata.epoch,
            step=loaded.metadata.step,
            optimizer_step=loaded.metadata.optimizer_step,
            checkpoint=str(loaded.path),
        )

    def _should_step(self, epoch_step: int, *, total_batches: int | None) -> bool:
        return epoch_step % int(self.config.gradient_accumulation_steps) == 0 or (total_batches is not None and epoch_step == total_batches)

    def _epoch_step_limit(self, dataloader: Any) -> int:
        if self.config.steps_per_epoch is not None:
            return int(self.config.steps_per_epoch)
        if hasattr(dataloader, "__len__"):
            length = int(len(dataloader))
            if length <= 0:
                raise ValueError("Training dataloader is empty.")
            return length
        raise ValueError("steps_per_epoch is required for dataloaders without a finite length.")

    def _epoch_batches(self, dataloader: Any, steps_per_epoch: int):
        iterator = iter(dataloader)
        produced = 0
        while produced < steps_per_epoch:
            try:
                batch = next(iterator)
            except StopIteration:
                if produced == 0:
                    raise ValueError("Training dataloader is empty.") from None
                iterator = iter(dataloader)
                continue
            produced += 1
            yield produced, batch

    def _planned_total_steps(self, dataloader: Any) -> int:
        total = int(self.config.epochs) * self._epoch_step_limit(dataloader)
        if self.config.max_steps is not None:
            total = min(total, int(self.config.max_steps))
        return total

    def _progress(self, planned_steps: int) -> float:
        if planned_steps <= 0:
            return 0.0
        return min(1.0, float(self.global_step) / float(planned_steps))

    def _early_stopping_metric(self, *, validation: float | None, train_losses: list[float]) -> float | None:
        metric = str(self.config.early_stopping_metric or "validation_loss").lower().strip()
        if metric == "validation_loss":
            if validation is not None:
                return float(validation)
            if train_losses:
                return sum(train_losses) / len(train_losses)
            return None
        if metric == "train_loss":
            return sum(train_losses) / len(train_losses) if train_losses else None
        raise ValueError("early_stopping_metric must be one of: validation_loss, train_loss.")


    def _save_checkpoint_if_due(self, *, model: Any, optimizer: Any, scheduler: Any | None, epoch: int, strategy: TrainingStrategy, metrics: dict[str, Any]) -> Path | None:
        if self.checkpoints is None or not self.config.checkpoint_interval:
            return None
        if self.global_step <= 0 or self.global_step % int(self.config.checkpoint_interval) != 0:
            return None
        return self._save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, strategy=strategy, metrics=metrics)

    def _save_epoch_checkpoint(self, *, model: Any, optimizer: Any, scheduler: Any | None, epoch: int, strategy: TrainingStrategy, metrics: dict[str, Any]) -> Path | None:
        if self.checkpoints is None:
            return None
        if self.config.checkpoint_interval:
            return None
        return self._save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, strategy=strategy, metrics=metrics)

    def _save_checkpoint(self, *, model: Any, optimizer: Any, scheduler: Any | None, epoch: int, strategy: TrainingStrategy, metrics: dict[str, Any]) -> Path:
        assert self.checkpoints is not None
        path = self.checkpoints.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=self.global_step,
            optimizer_step=self.optimizer_step,
            strategy=strategy.name,
            metrics=metrics,
        )
        self._emit("checkpoint", epoch=epoch, checkpoint=str(path), metrics=metrics)
        return path

    def _validate_if_due(self, *, model: Any, validation_loader: Any | None, strategy: TrainingStrategy, epoch: int, elapsed: float) -> float | None:
        if validation_loader is None:
            return None
        if self.config.validation_interval is None:
            return None
        if epoch % int(self.config.validation_interval) != 0:
            return None
        report = self.evaluator.evaluate(model=model, dataloader=validation_loader, strategy=strategy, engine=self)
        self._emit("validation_end", epoch=epoch, metrics=report.to_dict(), elapsed=elapsed)
        return report.loss

    def _emit(self, name: str, *, epoch: int = 0, checkpoint: str | None = None, metrics: dict[str, Any] | None = None, elapsed: float = 0.0) -> None:
        self.callbacks.emit(
            TrainingEvent(
                name=name,
                epoch=epoch,
                step=self.global_step,
                optimizer_step=self.optimizer_step,
                metrics=dict(metrics or {}),
                checkpoint=checkpoint,
                elapsed_time=elapsed,
            )
        )


__all__ = ["ResumeState", "TrainingCancelled", "TrainingEngine"]
