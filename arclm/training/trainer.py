"""Public training orchestration API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..datasets import Dataset
from ..models import Model
from .checkpoints import CheckpointManager
from .config import FineTuningConfig, TrainingConfig, TrainingPlan
from .engine import TrainingEngine
from .strategies import AdapterStrategy, FullFineTuneStrategy, PretrainStrategy


class Trainer:
    """Public ArcLM trainer over the owned training engine."""

    def __init__(self, *args: Any, model: Model | None = None, dataset: Dataset | None = None, **kwargs: Any):
        if args:
            model = args[0]
        if len(args) > 1:
            dataset = args[1]
        if model is None or dataset is None:
            raise TypeError("Trainer(model=..., dataset=...) requires a Model and Dataset.")

        self.model = model
        self.dataset = dataset
        config = kwargs.pop("config", None)
        fine_tuning = kwargs.pop("fine_tuning", None)
        self.strategy_override = kwargs.pop("strategy", None)
        self.validation_dataset = kwargs.pop("validation_dataset", None)
        self.callbacks = list(kwargs.pop("callbacks", []) or [])
        self.fine_tuning = self._fine_tuning_config(kwargs, fine_tuning=fine_tuning)
        self.config = self._training_config(model, kwargs, config=config)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unknown Trainer option(s): {unknown}")
        self.plan = self.make_plan()
        self.history: dict[str, Any] = {}

    def make_plan(self) -> TrainingPlan:
        """Create an inspectable plan without executing training."""

        return TrainingPlan(
            strategy=getattr(self.strategy_override, "name", None) or self.fine_tuning.normalized_method(),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            block_size=self.config.block_size,
            runtime=self.model.runtime.to_dict(),
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            steps_per_epoch=self.config.steps_per_epoch,
            checkpoint_interval=self.config.checkpoint_interval,
            validation_interval=self.config.validation_interval,
            max_steps=self.config.max_steps,
            early_stopping={
                "enabled": self.config.early_stopping,
                "patience": self.config.early_stopping_patience,
                "min_delta": self.config.early_stopping_min_delta,
                "metric": self.config.early_stopping_metric,
            },
            fine_tuning=self.fine_tuning.to_dict(),
        )

    def inspect(self) -> dict[str, Any]:
        """Return current trainer state."""

        return {"plan": self.plan.to_dict(), "history": dict(self.history)}

    def memory_plan(self) -> dict[str, Any]:
        """Report model/runtime/fine-tuning memory planning before training."""

        method = self.fine_tuning.normalized_method()
        if method == "adapter":
            method = self.fine_tuning.adapter_type
        report = self.model.memory_plan(fine_tuning=method)
        report["training"] = {
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "block_size": self.config.block_size,
            "steps_per_epoch": self.config.steps_per_epoch,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
        }
        report["fine_tuning"] = self.fine_tuning.to_dict()
        return report

    def train(self, *args: Any, mode: str | None = None, debug: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Train or fine-tune using ArcLM's training engine."""

        if args:
            raise TypeError("Trainer.train() accepts keyword arguments only.")
        optimizer = kwargs.pop("optimizer", None)
        scheduler = kwargs.pop("scheduler", None)
        if kwargs:
            self._apply_train_overrides(kwargs)

        self.model.config.batch_size = self.config.batch_size
        self.model.config.learning_rate = self.config.learning_rate
        self.model.config.num_epochs = self.config.epochs

        strategy = self._strategy(mode or self.strategy_override or self.fine_tuning.normalized_method())
        prepared = self.dataset.prepare(
            tokenizer=self.model.tokenizer,
            max_vocab=int(getattr(self.model.config, "max_vocab", 50000) or 50000),
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
        )
        self.model.tokenizer = prepared.tokenizer
        self.model.config.vocab_size = prepared.tokenizer.get_vocab_size()
        validation_loader = self._prepare_validation_loader(prepared.tokenizer)

        checkpoint_manager = CheckpointManager(self.config.checkpoint_dir) if self.config.checkpoint_dir else None
        engine = TrainingEngine(
            runtime=self.model.runtime,
            config=self.config,
            callbacks=self.callbacks,
            checkpoint_manager=checkpoint_manager,
        )
        result = engine.fit(
            model=self.model.model,
            dataloader=prepared.train_loader,
            validation_loader=validation_loader,
            strategy=strategy,
            optimizer=optimizer,
            scheduler=scheduler,
            debug=debug,
        )

        if self.config.save_artifact:
            artifact_path = self.config.artifact_path or Path("model.arcmodel")
            artifact = self.model.save(
                artifact_path,
                overwrite=True,
                metadata={"training": {"plan": self.plan.to_dict(), "history": result.to_history()}},
            )
            result.artifact = str(artifact.path)
        if self.config.save_adapter:
            adapter_path = self.config.adapter_path or Path("adapter.arcadapter")
            adapter = self.model.save_adapter(adapter_path, overwrite=True)
            result.adapter = str(Path(adapter_path).with_suffix(".arcadapter"))

        self.history = result.to_history()
        return self.history

    def fine_tune(self, *args: Any, method: str | None = None, debug: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Fine-tune through the same ArcLM training engine."""

        mode = method or self.fine_tuning.normalized_method()
        if mode in {"lora", "peft"}:
            mode = "adapter"
        return self.train(*args, mode=mode, debug=debug, **kwargs)

    def _prepare_validation_loader(self, tokenizer: Any) -> Any | None:
        if self.validation_dataset is None:
            return None
        dataset = self.validation_dataset if isinstance(self.validation_dataset, Dataset) else Dataset.load(self.validation_dataset)
        prepared = dataset.prepare(
            tokenizer=tokenizer,
            max_vocab=int(getattr(self.model.config, "max_vocab", 50000) or 50000),
            block_size=self.config.block_size,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        return prepared.train_loader

    def _strategy(self, mode: Any) -> Any:
        if hasattr(mode, "prepare") and hasattr(mode, "loss"):
            return mode
        normalized = str(mode or "pretrain").lower().strip().replace("-", "_")
        if normalized == "pretrain":
            return PretrainStrategy()
        if normalized in {"full", "full_finetune", "finetune"}:
            return FullFineTuneStrategy(self.fine_tuning)
        if normalized in {"adapter", "lora", "peft"}:
            self._ensure_lora_attached()
            return AdapterStrategy(self.fine_tuning)
        raise ValueError("mode must be one of: pretrain, full_finetune, adapter, lora.")

    def _ensure_lora_attached(self) -> None:
        if any(".lora_" in name for name, _ in self.model.model.named_parameters()):
            return
        self.model.attach_lora(
            rank=self.fine_tuning.rank,
            alpha=self.fine_tuning.alpha,
            target_modules=self.fine_tuning.target_modules or None,
        )

    @staticmethod
    def _fine_tuning_config(options: dict[str, Any], *, fine_tuning: FineTuningConfig | dict[str, Any] | None = None) -> FineTuningConfig:
        if isinstance(fine_tuning, FineTuningConfig):
            base = fine_tuning.to_dict()
        elif isinstance(fine_tuning, dict):
            base = dict(fine_tuning)
        else:
            base = {}
        method = options.pop("fine_tuning_method", options.pop("method", base.get("method", "pretrain")))
        return FineTuningConfig(
            method=method,
            freeze_base_model=bool(options.pop("freeze_base_model", base.get("freeze_base_model", False))),
            trainable_patterns=tuple(options.pop("trainable_patterns", base.get("trainable_patterns", ())) or ()),
            freeze_patterns=tuple(options.pop("freeze_patterns", base.get("freeze_patterns", ())) or ()),
            adapter_type=str(options.pop("adapter_type", base.get("adapter_type", "lora"))),
            rank=int(options.pop("rank", base.get("rank", 4))),
            alpha=float(options.pop("alpha", base.get("alpha", 8.0))),
            target_modules=tuple(options.pop("target_modules", base.get("target_modules", ())) or ()),
        )

    @staticmethod
    def _training_config(model: Model, options: dict[str, Any], *, config: TrainingConfig | dict[str, Any] | None = None) -> TrainingConfig:
        if isinstance(config, TrainingConfig):
            base = config.to_dict()
        elif isinstance(config, dict):
            base = dict(config)
        else:
            base = {}
        grad_clip = options.pop("grad_clip", None)
        checkpoint_dir = options.pop("checkpoint_dir", None)
        resume_from = options.pop("resume_from", None)
        artifact_path = options.pop("artifact_path", None)
        adapter_path = options.pop("adapter_path", None)
        steps = options.pop("steps", base.get("steps"))
        steps_per_epoch = options.pop("steps_per_epoch", steps)
        early_stopping_patience = options.pop("early_stopping_patience", None)
        if grad_clip is None:
            grad_clip = base.get("grad_clip")
        if checkpoint_dir is None:
            checkpoint_dir = base.get("checkpoint_dir")
        if resume_from is None:
            resume_from = base.get("resume_from")
        if artifact_path is None:
            artifact_path = base.get("artifact_path")
        if adapter_path is None:
            adapter_path = base.get("adapter_path")
        if steps_per_epoch is None:
            steps_per_epoch = base.get("steps_per_epoch")
        if early_stopping_patience is None:
            early_stopping_patience = base.get("early_stopping_patience")
        return TrainingConfig(
            epochs=int(options.pop("epochs", base.get("epochs", getattr(model.config, "num_epochs", 1) or 1))),
            batch_size=int(options.pop("batch_size", base.get("batch_size", getattr(model.config, "batch_size", 2) or 2))),
            learning_rate=float(options.pop("learning_rate", base.get("learning_rate", getattr(model.config, "learning_rate", 1e-3) or 1e-3))),
            block_size=int(options.pop("block_size", base.get("block_size", getattr(model.config, "block_size", 8) or 8))),
            weight_decay=float(options.pop("weight_decay", base.get("weight_decay", getattr(model.config, "weight_decay", 0.0) or 0.0))),
            grad_clip=float(grad_clip) if grad_clip is not None else None,
            gradient_accumulation_steps=int(options.pop("gradient_accumulation_steps", base.get("gradient_accumulation_steps", 1))),
            steps_per_epoch=int(steps_per_epoch) if steps_per_epoch is not None else None,
            checkpoint_interval=options.pop("checkpoint_interval", base.get("checkpoint_interval")),
            validation_interval=options.pop("validation_interval", base.get("validation_interval", 1)),
            log_interval=options.pop("log_interval", base.get("log_interval")),
            max_steps=options.pop("max_steps", base.get("max_steps")),
            early_stopping=bool(options.pop("early_stopping", base.get("early_stopping", False))),
            early_stopping_patience=int(early_stopping_patience) if early_stopping_patience is not None else None,
            early_stopping_min_delta=float(options.pop("early_stopping_min_delta", base.get("early_stopping_min_delta", 0.0))),
            early_stopping_metric=str(options.pop("early_stopping_metric", base.get("early_stopping_metric", "validation_loss"))),
            show_progress=bool(options.pop("show_progress", base.get("show_progress", False))),
            shuffle=bool(options.pop("shuffle", base.get("shuffle", True))),
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir is not None else None,
            resume_from=Path(resume_from) if resume_from is not None else None,
            save_artifact=bool(options.pop("save_artifact", base.get("save_artifact", False))),
            artifact_path=Path(artifact_path) if artifact_path is not None else None,
            save_adapter=bool(options.pop("save_adapter", base.get("save_adapter", False))),
            adapter_path=Path(adapter_path) if adapter_path is not None else None,
            metadata=dict(base.get("metadata", {})),
        )

    def _apply_train_overrides(self, options: dict[str, Any]) -> None:
        if "steps" in options and "steps_per_epoch" not in options:
            options["steps_per_epoch"] = options.pop("steps")
        override = {**self.config.to_dict(), **options}
        for key in ("checkpoint_dir", "resume_from", "artifact_path", "adapter_path"):
            if override.get(key) is not None:
                override[key] = Path(override[key])
        self.config = TrainingConfig(**override)
        self.plan = self.make_plan()


__all__ = ["Trainer", "TrainingPlan"]
