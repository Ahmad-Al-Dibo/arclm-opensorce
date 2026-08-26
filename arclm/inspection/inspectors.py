"""Structured inspectors for ArcLM objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InspectorReport:
    """Small report that can feed CLI or web surfaces later."""

    kind: str
    summary: dict[str, Any]
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelInspector:
    def inspect(self, model: Any) -> InspectorReport:
        details = model.inspect() if hasattr(model, "inspect") else {}
        summary = {
            "architecture_id": details.get("architecture_id"),
            "parameters": details.get("parameters"),
            "tokenizer": details.get("tokenizer"),
            "base_model_id": details.get("base_model_id"),
        }
        return InspectorReport(kind="model", summary=summary, details=details)


class DatasetInspector:
    def inspect(self, dataset: Any) -> InspectorReport:
        report = dataset.inspect()
        details = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        summary = {
            "source": details.get("source"),
            "records": details.get("records"),
            "fields": details.get("fields", []),
            "estimated_tokens": details.get("estimated_tokens"),
        }
        return InspectorReport(kind="dataset", summary=summary, details=details, warnings=list(details.get("warnings", [])))


class TrainingInspector:
    def inspect(self, trainer_or_history: Any) -> InspectorReport:
        if isinstance(trainer_or_history, dict):
            history = dict(trainer_or_history)
            plan = {}
        else:
            payload = trainer_or_history.inspect() if hasattr(trainer_or_history, "inspect") else {}
            plan = dict(payload.get("plan", {}))
            history = dict(payload.get("history", {}))
        summary = {
            "strategy": history.get("strategy") or plan.get("strategy"),
            "global_steps": history.get("global_steps"),
            "training_loss": self._last(history.get("train_losses")),
            "validation_loss": self._last(history.get("validation_losses")),
            "elapsed_time": history.get("elapsed_time"),
        }
        return InspectorReport(kind="training", summary=summary, details={"plan": plan, "history": history})

    @staticmethod
    def _last(values: Any) -> Any:
        return values[-1] if isinstance(values, list) and values else None


class RuntimeInspector:
    def inspect(self, runtime: Any) -> InspectorReport:
        details = runtime.to_dict() if hasattr(runtime, "to_dict") else {}
        summary = {
            "backend": details.get("backend"),
            "device": (details.get("device") or {}).get("type"),
            "precision": details.get("precision"),
        }
        return InspectorReport(kind="runtime", summary=summary, details=details, warnings=list(details.get("warnings", [])))


class ArtifactInspector:
    def inspect(self, artifact: Any) -> InspectorReport:
        from ..artifacts import ArcModelArtifact

        active = ArcModelArtifact(artifact) if isinstance(artifact, (str, Path)) else artifact
        details = active.inspect() if hasattr(active, "inspect") else {}
        summary = {
            "path": details.get("path") or str(getattr(active, "path", "")),
            "format": details.get("format"),
            "architecture_id": details.get("architecture_id"),
            "layout": details.get("layout"),
        }
        return InspectorReport(kind="artifact", summary=summary, details=details)


__all__ = ["ArtifactInspector", "DatasetInspector", "InspectorReport", "ModelInspector", "RuntimeInspector", "TrainingInspector"]
