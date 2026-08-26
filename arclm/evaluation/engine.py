"""ArcLM evaluation engine for experiments and reports."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class EvaluationContext:
    """Inputs available to evaluation plug-ins."""

    model: Any | None = None
    dataset: Any | None = None
    history: dict[str, Any] = field(default_factory=dict)
    runtime: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricEvaluator(Protocol):
    """Research contract for custom evaluators."""

    name: str

    def evaluate(self, context: EvaluationContext) -> dict[str, Any] | float | int: ...


@dataclass(frozen=True)
class EvaluationReport:
    """Structured evaluation output."""

    metrics: dict[str, Any]
    custom_metrics: dict[str, Any] = field(default_factory=dict)
    task_reports: dict[str, Any] = field(default_factory=dict)
    generation_reports: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvaluationEngine:
    """Create reusable evaluation reports from training and task outputs."""

    def evaluate(
        self,
        *,
        model: Any | None = None,
        dataset: Any | None = None,
        trainer: Any | None = None,
        history: dict[str, Any] | None = None,
        evaluators: list[Any] | tuple[Any, ...] = (),
        generation_prompts: list[str] | tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        started = time.monotonic()
        active_history = dict(history or getattr(trainer, "history", {}) or {})
        context = EvaluationContext(
            model=model or getattr(trainer, "model", None),
            dataset=dataset or getattr(trainer, "dataset", None),
            history=active_history,
            runtime=getattr(model or getattr(trainer, "model", None), "runtime", None),
            metadata=dict(metadata or {}),
        )
        metrics = self._history_metrics(active_history)
        custom_metrics = self._custom_metrics(context, evaluators)
        generation_reports = self._generation_reports(context, generation_prompts)
        metrics["evaluation_time"] = time.monotonic() - started
        return EvaluationReport(
            metrics=metrics,
            custom_metrics=custom_metrics,
            generation_reports=generation_reports,
            metadata={
                "model": self._model_label(context.model),
                "dataset": self._dataset_label(context.dataset),
                **dict(metadata or {}),
            },
        )

    @staticmethod
    def _history_metrics(history: dict[str, Any]) -> dict[str, Any]:
        train_losses = [float(value) for value in history.get("train_losses", []) if value is not None]
        validation_losses = [float(value) for value in history.get("validation_losses", []) if value is not None]
        loss = validation_losses[-1] if validation_losses else (train_losses[-1] if train_losses else None)
        elapsed = float(history.get("elapsed_time") or 0.0)
        tokens = int(history.get("tokens_processed") or 0)
        return {
            "training_loss": train_losses[-1] if train_losses else None,
            "training_loss_avg": sum(train_losses) / len(train_losses) if train_losses else None,
            "validation_loss": validation_losses[-1] if validation_losses else None,
            "validation_loss_avg": sum(validation_losses) / len(validation_losses) if validation_losses else None,
            "loss": loss,
            "perplexity": math.exp(loss) if loss is not None and loss < 100 else None,
            "tokens_processed": tokens,
            "samples_processed": int(history.get("samples_processed") or 0),
            "global_steps": int(history.get("global_steps") or 0),
            "elapsed_time": elapsed,
            "tokens_per_second": tokens / elapsed if elapsed > 0 and tokens else None,
        }

    @staticmethod
    def _custom_metrics(context: EvaluationContext, evaluators: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for evaluator in evaluators:
            name = str(getattr(evaluator, "name", evaluator.__class__.__name__))
            if hasattr(evaluator, "evaluate"):
                value = evaluator.evaluate(context)
            else:
                value = evaluator(context)
            metrics[name] = value
        return metrics

    @staticmethod
    def _generation_reports(context: EvaluationContext, prompts: list[str] | tuple[str, ...]) -> dict[str, Any]:
        if not prompts or context.model is None or not hasattr(context.model, "generate"):
            return {}
        reports = {}
        for prompt in prompts:
            try:
                reports[prompt] = context.model.generate(prompt, max_new_tokens=8)
            except Exception as exc:
                reports[prompt] = {"error": repr(exc)}
        return reports

    @staticmethod
    def _model_label(model: Any | None) -> str | None:
        if model is None:
            return None
        return getattr(model, "base_model_id", None) or getattr(getattr(model, "architecture", None), "architecture_id", None)

    @staticmethod
    def _dataset_label(dataset: Any | None) -> str | None:
        if dataset is None:
            return None
        return getattr(dataset, "source", None)


__all__ = ["EvaluationContext", "EvaluationEngine", "EvaluationReport", "MetricEvaluator"]
