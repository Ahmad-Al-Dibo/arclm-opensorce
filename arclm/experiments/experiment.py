"""Lightweight local ArcLM experiment tracking."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._version import __version__
from ..evaluation import EvaluationEngine, EvaluationReport


@dataclass(frozen=True)
class ExperimentComparison:
    """Comparison table for multiple experiments."""

    experiments: list[str]
    rows: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Experiment:
    """Local experiment record stored as JSON."""

    FILENAME = "experiment.json"

    def __init__(self, name: str, *, root: str | Path = ".arclm/experiments", seed: int | None = None):
        self.name = str(name)
        self.root = Path(root)
        self.seed = int(seed if seed is not None else 0)
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.record: dict[str, Any] = {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "seed": self.seed,
            "arclm_version": __version__,
            "model": None,
            "dataset": None,
            "configuration": {},
            "runtime": None,
            "training_history": {},
            "evaluation": {},
            "artifacts": {},
            "reproducibility": {},
        }

    @property
    def path(self) -> Path:
        return self.root / self.name / self.FILENAME

    def capture(
        self,
        *,
        model: Any | None = None,
        dataset: Any | None = None,
        trainer: Any | None = None,
        history: dict[str, Any] | None = None,
        evaluation: EvaluationReport | dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> "Experiment":
        """Capture experiment metadata from ArcLM objects."""

        if seed is not None:
            self.seed = int(seed)
            self.record["seed"] = self.seed
        active_history = dict(history or getattr(trainer, "history", {}) or {})
        if evaluation is None:
            evaluation = EvaluationEngine().evaluate(model=model or getattr(trainer, "model", None), dataset=dataset or getattr(trainer, "dataset", None), history=active_history)
        evaluation_payload = evaluation.to_dict() if hasattr(evaluation, "to_dict") else dict(evaluation)
        model_report = self._model_report(model or getattr(trainer, "model", None))
        dataset_report = self._dataset_report(dataset or getattr(trainer, "dataset", None))
        runtime = model_report.get("runtime") if model_report else None
        trainer_plan = trainer.make_plan().to_dict() if trainer is not None and hasattr(trainer, "make_plan") else {}
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.record.update(
            {
                "updated_at": self.updated_at,
                "model": model_report,
                "dataset": dataset_report,
                "configuration": {**trainer_plan, **dict(configuration or {})},
                "runtime": runtime,
                "training_history": active_history,
                "evaluation": evaluation_payload,
                "artifacts": self._artifacts(active_history, artifacts),
                "reproducibility": {
                    "seed": self.seed,
                    "python_random_state": self._seed_preview(self.seed),
                    "model_fingerprint": (model_report or {}).get("base_model_fingerprint"),
                    "dataset_fingerprint": (dataset_report or {}).get("fingerprint"),
                    "arclm_version": __version__,
                    "architecture_version": (model_report or {}).get("architecture_version"),
                    "training_configuration": trainer_plan,
                },
            }
        )
        return self

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.record, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return self.path

    def to_dict(self) -> dict[str, Any]:
        return dict(self.record)

    @classmethod
    def load(cls, path_or_name: str | Path, *, root: str | Path = ".arclm/experiments") -> "Experiment":
        path = Path(path_or_name)
        if not path.exists():
            path = Path(root) / str(path_or_name) / cls.FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        experiment = cls(data["name"], root=path.parent.parent, seed=data.get("seed"))
        experiment.record = data
        experiment.created_at = data.get("created_at", experiment.created_at)
        experiment.updated_at = data.get("updated_at", experiment.updated_at)
        return experiment

    @classmethod
    def compare(cls, *experiments: "Experiment" | str | Path, root: str | Path = ".arclm/experiments") -> ExperimentComparison:
        loaded = [item if isinstance(item, Experiment) else cls.load(item, root=root) for item in experiments]
        rows = [cls._comparison_row(experiment.to_dict()) for experiment in loaded]
        return ExperimentComparison(experiments=[experiment.name for experiment in loaded], rows=rows)

    @staticmethod
    def _comparison_row(record: dict[str, Any]) -> dict[str, Any]:
        metrics = ((record.get("evaluation") or {}).get("metrics") or {})
        history = record.get("training_history") or {}
        trainable = ((history.get("metrics") or {}).get("trainable") or {})
        memory = ((record.get("model") or {}).get("memory_plan") or {})
        return {
            "name": record.get("name"),
            "loss": metrics.get("loss"),
            "training_loss": metrics.get("training_loss"),
            "validation_loss": metrics.get("validation_loss"),
            "perplexity": metrics.get("perplexity"),
            "elapsed_time": metrics.get("elapsed_time"),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "trainable_parameters": trainable.get("trainable_parameters") or memory.get("trainable_parameters"),
            "estimated_parameter_memory_bytes": memory.get("estimated_parameter_memory_bytes"),
            "artifact": (record.get("artifacts") or {}).get("model"),
            "adapter": (record.get("artifacts") or {}).get("adapter"),
        }

    @staticmethod
    def _model_report(model: Any | None) -> dict[str, Any] | None:
        if model is None:
            return None
        report = model.inspect() if hasattr(model, "inspect") else {}
        if hasattr(model, "memory_plan"):
            report["memory_plan"] = model.memory_plan()
        return report

    @staticmethod
    def _dataset_report(dataset: Any | None) -> dict[str, Any] | None:
        if dataset is None:
            return None
        inspection = dataset.inspect()
        payload = inspection.to_dict() if hasattr(inspection, "to_dict") else dict(inspection)
        text = dataset.text() if hasattr(dataset, "text") else json.dumps(payload, sort_keys=True)
        payload["fingerprint"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return payload

    @staticmethod
    def _artifacts(history: dict[str, Any], artifacts: dict[str, Any] | None) -> dict[str, Any]:
        payload = {key: str(value) for key, value in dict(artifacts or {}).items()}
        if history.get("artifact"):
            payload.setdefault("model", history["artifact"])
        if history.get("adapter"):
            payload.setdefault("adapter", history["adapter"])
        if history.get("checkpoints"):
            payload.setdefault("checkpoints", list(history["checkpoints"]))
        return payload

    @staticmethod
    def _seed_preview(seed: int) -> float:
        rng = random.Random(seed)
        return rng.random()


__all__ = ["Experiment", "ExperimentComparison"]
