"""Beginner-friendly Lab API over the shared ArcLM core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..runtime import Runtime
from .dataset import Dataset, DatasetInspection
from .model import Model
from .trainer import Trainer, TrainingPlan


@dataclass
class Lab:
    """High-level educational ArcLM interface."""

    runtime: Runtime = field(default_factory=Runtime.auto)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    last_dataset: Dataset | None = None
    last_model: Model | None = None
    last_trainer: Trainer | None = None

    def inspect(self, target: Any | None = None) -> dict[str, Any] | DatasetInspection:
        """Inspect data or the latest Lab decisions."""

        if target is None:
            return {
                "runtime": self.runtime.to_dict(),
                "decisions": list(self.decisions),
                "dataset": self.last_dataset.inspect().to_dict() if self.last_dataset else None,
                "model": self.last_model.inspect() if self.last_model else None,
                "trainer": self.last_trainer.inspect() if self.last_trainer else None,
            }
        dataset = Dataset.load(target)
        self.last_dataset = dataset
        report = dataset.inspect()
        self.decisions.append({"stage": "dataset.inspect", "report": report.to_dict()})
        return report

    def create(self, *, task: str = "causal-lm", size: str = "small", data: Dataset | None = None, **overrides: Any) -> Model:
        """Create a small native model with inspectable defaults."""

        if task != "causal-lm":
            raise ValueError("Lab.create() currently supports task='causal-lm' only.")
        sizes = {
            "tiny": {"embed_dim": 8, "block_size": 4, "num_blocks": 1, "batch_size": 2},
            "small": {"embed_dim": 16, "block_size": 8, "num_blocks": 1, "batch_size": 2},
        }
        config = {**sizes.get(size, sizes["small"]), **overrides}
        dataset = data or self.last_dataset
        tokenizer = None
        if dataset is not None:
            prepared = dataset.prepare(block_size=int(config["block_size"]), batch_size=int(config["batch_size"]))
            tokenizer = prepared.tokenizer
        elif "vocab_size" not in config:
            config["vocab_size"] = 256
        model = Model.create(architecture="arclm-native", tokenizer=tokenizer, runtime=self.runtime, **config)
        self.last_model = model
        self.decisions.append({"stage": "model.create", "size": size, "config": model.config.to_dict()})
        return model

    def plan(self, model: Model, data: Dataset, **options: Any) -> TrainingPlan:
        """Create a training plan."""

        trainer = Trainer(model=model, dataset=data, **options)
        self.last_trainer = trainer
        assert trainer.plan is not None
        self.decisions.append({"stage": "training.plan", "plan": trainer.plan.to_dict()})
        return trainer.plan

    def train(self, model: Model, data: Dataset, **options: Any) -> dict[str, Any]:
        """Train a model through the shared high-level Trainer."""

        trainer = Trainer(model=model, dataset=data, **options)
        self.last_trainer = trainer
        history = trainer.train()
        self.decisions.append({"stage": "training.execute", "history": history})
        return history

    def pretrain(self, data: str | Dataset, *, size: str = "small", **options: Any) -> Model:
        """Inspect data, create a model, train it, and return the model."""

        dataset = data if isinstance(data, Dataset) else Dataset.load(data)
        self.last_dataset = dataset
        model = self.create(size=size, data=dataset, **{key: value for key, value in options.items() if key in {"embed_dim", "block_size", "num_blocks", "batch_size", "vocab_size"}})
        train_options = {key: value for key, value in options.items() if key in {"epochs", "batch_size", "learning_rate", "block_size"}}
        self.train(model, dataset, **train_options)
        return model
