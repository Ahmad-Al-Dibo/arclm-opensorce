"""Unified ArcLM-facing training engine."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .._version import __version__
from ..events import CallbackManager, Event, EventHandler
from ..exceptions import ConfigurationError, TrainingError


@dataclass(frozen=True)
class TrainingConfig:
    """Validated training configuration used by :func:`train`."""

    output_dir: str = "runs/training"
    task: str = "sft"
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    max_grad_norm: Optional[float] = 1.0
    evaluation_strategy: str = "none"
    evaluation_steps: Optional[int] = None
    checkpoint_strategy: str = "none"
    checkpoint_steps: Optional[int] = None
    max_steps: Optional[int] = None
    seed: int = 42
    optimizer: str = "adamw"
    scheduler: str = "constant"
    precision: str = "auto"
    device: str = "auto"
    early_stopping_patience: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    save_tokenizer: bool = True

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ConfigurationError("epochs must be greater than zero.")
        if self.batch_size <= 0:
            raise ConfigurationError("batch_size must be greater than zero.")
        if self.gradient_accumulation_steps <= 0:
            raise ConfigurationError("gradient_accumulation_steps must be greater than zero.")
        if not 0 < self.learning_rate <= 1:
            raise ConfigurationError("learning_rate must be in the interval (0, 1].")
        if self.weight_decay < 0:
            raise ConfigurationError("weight_decay must be non-negative.")
        if not 0 <= self.warmup_ratio <= 1:
            raise ConfigurationError("warmup_ratio must be in the interval [0, 1].")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ConfigurationError("max_grad_norm must be positive when set.")
        if self.evaluation_strategy not in {"none", "steps", "epoch"}:
            raise ConfigurationError("evaluation_strategy must be 'none', 'steps', or 'epoch'.")
        if self.checkpoint_strategy not in {"none", "steps", "epoch"}:
            raise ConfigurationError("checkpoint_strategy must be 'none', 'steps', or 'epoch'.")
        if self.evaluation_strategy == "steps" and not self.evaluation_steps:
            raise ConfigurationError("evaluation_steps is required when evaluation_strategy='steps'.")
        if self.checkpoint_strategy == "steps" and not self.checkpoint_steps:
            raise ConfigurationError("checkpoint_steps is required when checkpoint_strategy='steps'.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ConfigurationError("max_steps must be positive when set.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingReport:
    """Structured training result."""

    backend: str
    output_dir: str
    steps: int
    duration_seconds: float
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoints: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    report_type: str = "training_report"
    schema_version: str = "1.0"
    arclm_version: str = __version__
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train(
    *,
    model: Any = None,
    dataset: Any = None,
    config: TrainingConfig | None = None,
    callbacks: Optional[Iterable[EventHandler]] = None,
    model_name: Optional[str] = None,
) -> TrainingReport:
    """Train through ArcLM's unified facade.

    The current implementation delegates certified Hugging Face SFT to
    :func:`arclm.sft.train_sft`. Native checkpoint training remains available
    through ``arclm.pipeline.train_model`` and will be moved behind this facade
    in a later stabilization pass.
    """

    cfg = config or TrainingConfig()
    manager = CallbackManager(callbacks)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manager.emit(Event("run_started", {"task": cfg.task}))
    try:
        if cfg.task != "sft":
            raise TrainingError("Unified train currently implements task='sft' only.")
        data_path = _dataset_to_jsonl(dataset, output_dir / "train.jsonl")
        source = model_name or getattr(model, "source", None)
        if not source:
            raise TrainingError("A model source or model bundle with .source is required.")
        from ..sft import train_sft

        result = train_sft(
            model=source,
            dataset=str(data_path),
            output_dir=str(output_dir),
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            num_epochs=cfg.epochs,
            max_steps=cfg.max_steps,
            seed=cfg.seed,
            trust_remote_code=False,
            save_tokenizer=cfg.save_tokenizer,
        )
        manager.emit(Event("step_completed", {"steps": result.steps}, step=result.steps))
        manager.emit(Event("run_completed", {"output_dir": str(output_dir)}))
        report = TrainingReport(
            backend=result.backend,
            output_dir=str(output_dir),
            steps=result.steps,
            duration_seconds=time.perf_counter() - started,
            metrics={"loss_history": result.train_loss_history},
            checkpoints=[str(output_dir)],
            warnings=list(manager.warnings),
            config=cfg.to_dict(),
        )
        (output_dir / "training-report.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return report
    except Exception as exc:
        manager.emit(Event("run_failed", {"error": str(exc)}))
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(str(exc)) from exc


def _dataset_to_jsonl(dataset: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(dataset, (str, Path)):
        return Path(dataset)
    records = getattr(dataset, "records", dataset)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            if "prompt" in row and "completion" in row:
                payload = {"prompt": row["prompt"], "completion": row["completion"]}
            elif "text" in row:
                text = str(row["text"])
                payload = {"prompt": text, "completion": text}
            else:
                payload = dict(row)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


__all__ = ["TrainingConfig", "TrainingReport", "train"]
