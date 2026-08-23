"""Unified public Trainer bridge for ArcLM vNext."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..pipeline import build_trainer
from ..trainer import Trainer as LegacyTrainer
from .dataset import Dataset
from .model import Model


@dataclass
class TrainingPlan:
    """Inspectable training plan."""

    epochs: int
    batch_size: int
    learning_rate: float
    block_size: int
    runtime: dict[str, Any]
    strategy: str = "pretrain"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe plan."""

        return {
            "strategy": self.strategy,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "block_size": self.block_size,
            "runtime": self.runtime,
        }


class Trainer:
    """Public Trainer that supports new high-level and legacy construction."""

    def __init__(self, *args: Any, model: Model | None = None, dataset: Dataset | None = None, **kwargs: Any):
        if args and not isinstance(args[0], Model):
            self._legacy = LegacyTrainer(*args, **kwargs)
            self.model = None
            self.dataset = None
            self.plan = None
            self.history: dict[str, Any] = {}
            return

        if args:
            model = args[0]
        if len(args) > 1:
            dataset = args[1]
        if model is None or dataset is None:
            raise TypeError("Trainer(model=..., dataset=...) requires a Model and Dataset.")

        self._legacy = None
        self.model = model
        self.dataset = dataset
        self.epochs = int(kwargs.pop("epochs", getattr(model.config, "num_epochs", 1) or 1))
        self.batch_size = int(kwargs.pop("batch_size", getattr(model.config, "batch_size", 2) or 2))
        self.learning_rate = float(kwargs.pop("learning_rate", getattr(model.config, "learning_rate", 1e-3) or 1e-3))
        self.block_size = int(kwargs.pop("block_size", getattr(model.config, "block_size", 8) or 8))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown Trainer option(s): {unknown}")
        self.plan = self.make_plan()
        self.history: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        legacy = self.__dict__.get("_legacy")
        if legacy is not None:
            return getattr(legacy, name)
        raise AttributeError(name)

    def make_plan(self) -> TrainingPlan:
        """Create an inspectable plan without executing training."""

        if self.model is None:
            raise RuntimeError("Legacy Trainer does not expose vNext training plans.")
        return TrainingPlan(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            block_size=self.block_size,
            runtime=self.model.runtime.to_dict(),
        )

    def inspect(self) -> dict[str, Any]:
        """Return current trainer state."""

        if self._legacy is not None:
            return self._legacy.get_train_history()
        return {"plan": self.plan.to_dict() if self.plan else None, "history": dict(self.history)}

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Train using the shared ArcLM training core."""

        if self._legacy is not None:
            return self._legacy.train(*args, **kwargs)
        if self.model is None or self.dataset is None:
            raise RuntimeError("Trainer is not initialized with model and dataset.")

        self.model.config.batch_size = self.batch_size
        self.model.config.learning_rate = self.learning_rate
        self.model.config.num_epochs = self.epochs
        self.model.config.block_size = self.block_size

        prepared = self.dataset.prepare(
            tokenizer=self.model.tokenizer,
            max_vocab=int(getattr(self.model.config, "max_vocab", 50000) or 50000),
            block_size=self.block_size,
            batch_size=self.batch_size,
        )
        self.model.tokenizer = prepared.tokenizer
        self.model.config.vocab_size = prepared.tokenizer.get_vocab_size()
        legacy = build_trainer(self.model.model, self.model.config)
        legacy.train(prepared.train_loader, self.epochs)
        self.history = legacy.get_train_history()
        return self.history
