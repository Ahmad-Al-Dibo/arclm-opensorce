"""Beginner-friendly Lab API over the shared ArcLM core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .datasets import Dataset, DatasetInspection
from .models import Model
from .runtime import Runtime
from .tokenizers import Tokenizer
from .training import Trainer, TrainingPlan


@dataclass
class Lab:
    """High-level educational ArcLM interface."""

    runtime: Runtime = field(default_factory=Runtime.auto)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    last_dataset: Dataset | None = None
    last_model: Model | None = None
    last_trainer: Trainer | None = None

    def dataset(
        self,
        source: str | Dataset | Any | None = None,
        *,
        text: str | None = None,
        records: Any | None = None,
        format: str | None = None,
    ) -> Dataset:
        """Load or create a dataset with beginner-friendly defaults."""

        if isinstance(source, Dataset):
            dataset = source
        elif text is not None:
            dataset = Dataset.load([{"text": text}], format="records")
        elif records is not None:
            dataset = Dataset.load(records, format=format or "records")
        elif source is not None:
            dataset = Dataset.load(source, format=format)
        else:
            raise TypeError("Lab.dataset(...) requires a path, Dataset, text=..., or records=....")
        self.last_dataset = dataset
        self.decisions.append({"stage": "student.dataset", "report": dataset.inspect().to_dict()})
        return dataset

    def model(self, *, data: Dataset | None = None, size: str = "tiny", tokenizer: Any | None = None, **options: Any) -> Model:
        """Create a model for the current or provided dataset."""

        return self.create(size=size, data=data or self.last_dataset, tokenizer=tokenizer, **options)

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

    def prepare(self, data: Dataset | None = None, **options: Any) -> Any:
        """Prepare the current or provided dataset with beginner-friendly defaults."""

        dataset = data or self.last_dataset
        if dataset is None:
            raise TypeError("Lab.prepare() requires a dataset or a previous Lab.dataset() call.")
        prepared = dataset.prepare(**options)
        self.last_dataset = dataset
        self.decisions.append(
            {
                "stage": "student.data.prepare",
                "records": len(dataset.records),
                "tokens": len(prepared.encoded),
                "block_size": prepared.block_size,
                "batch_size": prepared.batch_size,
            }
        )
        return prepared

    def create(self, *, task: str = "causal-lm", size: str = "small", data: Dataset | None = None, **overrides: Any) -> Model:
        """Create a small native model with inspectable defaults."""

        if task != "causal-lm":
            raise ValueError("Lab.create() currently supports task='causal-lm' only.")
        tokenizer_option = overrides.pop("tokenizer", None)
        sizes = {
            "tiny": {"embed_dim": 8, "block_size": 4, "num_blocks": 1, "batch_size": 2},
            "small": {"embed_dim": 16, "block_size": 8, "num_blocks": 1, "batch_size": 2},
        }
        config = {**sizes.get(size, sizes["small"]), **overrides}
        dataset = data or self.last_dataset
        tokenizer = self._coerce_tokenizer(tokenizer_option)
        if dataset is not None:
            prepared = dataset.prepare(tokenizer=tokenizer, block_size=int(config["block_size"]), batch_size=int(config["batch_size"]))
            tokenizer = prepared.tokenizer
        elif tokenizer is not None:
            config["vocab_size"] = tokenizer.get_vocab_size()
        elif "vocab_size" not in config:
            config["vocab_size"] = 256
        model = Model.create(architecture="arclm-native", tokenizer=tokenizer, runtime=self.runtime, **config)
        self.last_model = model
        self.decisions.append(
            {
                "stage": "model.create",
                "size": size,
                "tokenizer": self._tokenizer_label(tokenizer_option, tokenizer),
                "config": model.config.to_dict(),
            }
        )
        return model

    def plan(self, model: Model, data: Dataset, **options: Any) -> TrainingPlan:
        """Create a training plan."""

        trainer = Trainer(model=model, dataset=data, **options)
        self.last_trainer = trainer
        assert trainer.plan is not None
        self.decisions.append({"stage": "training.plan", "plan": trainer.plan.to_dict()})
        return trainer.plan

    def train(self, model: Model | None = None, data: Dataset | None = None, debug: bool = False, **options: Any) -> dict[str, Any]:
        """Train a model through the shared high-level Trainer."""

        active_model = model or self.last_model
        active_dataset = data or self.last_dataset
        if active_model is None or active_dataset is None:
            raise TypeError("Lab.train() requires a model and dataset, or previous Lab.model()/Lab.dataset() calls.")
        trainer = Trainer(model=active_model, dataset=active_dataset, **options)
        self.last_trainer = trainer
        history = trainer.train(debug=debug)
        self.decisions.append({"stage": "training.execute", "history": history})
        return history

    def fine_tune(self, model: Model | None = None, data: Dataset | None = None, *, method: str = "lora", debug: bool = False, **options: Any) -> dict[str, Any]:
        """Fine-tune a model through the same shared Trainer."""

        return self.train(model=model, data=data, debug=debug, method=method, **options)

    def pretrain(self, data: str | Dataset, *, size: str = "small", debug: bool = False, **options: Any) -> Model:
        """Inspect data, create a model, train it, and return the model."""

        dataset = data if isinstance(data, Dataset) else Dataset.load(data)
        self.last_dataset = dataset
        model = self.create(size=size, data=dataset, **{key: value for key, value in options.items() if key in {"embed_dim", "block_size", "num_blocks", "batch_size", "vocab_size", "tokenizer"}})
        train_options = {
            key: value
            for key, value in options.items()
            if key in {"epochs", "batch_size", "learning_rate", "block_size", "steps", "steps_per_epoch", "show_progress"}
        }

        self.train(model, dataset, debug=debug, **train_options)
        return model

    @staticmethod
    def _coerce_tokenizer(tokenizer: Any) -> Any:
        """Normalize Lab tokenizer options for the current native workflow."""

        if tokenizer is None:
            return None
        if hasattr(tokenizer, "get_vocab_size"):
            return tokenizer
        if isinstance(tokenizer, str):
            name = tokenizer.lower().strip()
            if name in {"word", "default", "auto"}:
                return None
            if name in {"word", "character", "char", "sentence", "sentencepiece"}:
                return None if name in {"word", "default", "auto"} else Tokenizer(strategy=name)
        raise TypeError("tokenizer must be an ArcLM Tokenizer object or one of: word, character, sentence.")

    @staticmethod
    def _tokenizer_label(requested: Any, tokenizer: Any) -> str | None:
        if requested is not None:
            return str(requested)
        if tokenizer is None:
            return None
        return type(tokenizer).__name__
