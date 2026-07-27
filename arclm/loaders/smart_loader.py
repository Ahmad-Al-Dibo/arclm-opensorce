"""Smart model-source inspection before external checkpoint loading."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .base import LoadedCheckpoint, load_torch_checkpoint
from .registry import load_external_model

logger = logging.getLogger(__name__)


@dataclass
class LoadPlan:
    """Detected and user-selected settings for loading a model source."""

    source: str
    source_type: str = "unknown"
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_type: Optional[str] = None
    architecture: Optional[str] = None
    tokenizer: Optional[str] = None
    weight_format: Optional[str] = None
    precision: Optional[str] = None
    load_as: Optional[str] = None
    has_optimizer: bool = False
    has_scheduler: bool = False
    has_training_state: bool = False
    can_resume_training: bool = False
    resume_epoch: Optional[int] = None
    resume_step: Optional[int] = None
    load_optimizer: Optional[bool] = None
    load_scheduler: Optional[bool] = None
    resume_training: Optional[bool] = None

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_type": self.source_type,
            "files": list(self.files),
            "metadata": dict(self.metadata),
            "model_type": self.model_type,
            "architecture": self.architecture,
            "tokenizer": self.tokenizer,
            "weight_format": self.weight_format,
            "precision": self.precision,
            "load_as": self.load_as,
            "has_optimizer": self.has_optimizer,
            "has_scheduler": self.has_scheduler,
            "has_training_state": self.has_training_state,
            "can_resume_training": self.can_resume_training,
            "resume_epoch": self.resume_epoch,
            "resume_step": self.resume_step,
            "load_optimizer": self.load_optimizer,
            "load_scheduler": self.load_scheduler,
            "resume_training": self.resume_training,
        }

    def format_report(self) -> str:
        model_name = self.metadata.get("name") or Path(self.source).name or self.source
        lines = [
            f"Model found: {model_name}",
            f"Architecture: {self.architecture or 'unknown'}",
            f"Tokenizer: {self.tokenizer or 'not found'}",
            f"Weights: {self.weight_format or 'not found'}",
            f"Precision: {(self.precision or 'unknown').upper()}",
            f"LoRA/PEFT adapter: {'present' if self.load_as == 'adapter' else 'not found'}",
            f"Optimizer state: {'found' if self.has_optimizer else 'not found'}",
            f"Scheduler state: {'found' if self.has_scheduler else 'not found'}",
        ]
        if self.can_resume_training:
            lines.append(
                "Training: can be resumed"
                f" from epoch {self.resume_epoch if self.resume_epoch is not None else 'unknown'},"
                f" step {self.resume_step if self.resume_step is not None else 'unknown'}"
            )
        else:
            lines.append("Training: cannot be resumed")
        return "\n".join(lines)

    @property
    def report(self) -> str:
        return self.format_report()


class ModelInspector:
    """Base class for source inspectors used by ``SmartLoader``."""

    def can_inspect(self, source: Any) -> bool:
        return True

    def inspect(self, source: Any) -> LoadPlan:
        return LoadPlan(source=str(source))


class DefaultModelInspector(ModelInspector):
    """Inspect local model folders, checkpoint files, and Hugging Face IDs."""

    weight_suffixes = {
        ".safetensors": "safetensors",
        ".bin": "bin",
        ".pth": "pth",
        ".pt": "pt",
        ".ckpt": "ckpt",
    }

    def inspect(self, source: Any) -> LoadPlan:
        source_str = str(source)
        path = Path(source_str)
        if path.is_dir():
            return self._inspect_dir(path)
        if path.is_file():
            return self._inspect_file(path)
        return self._inspect_remote_id(source_str)

    def _inspect_dir(self, path: Path) -> LoadPlan:
        files = sorted(item.name for item in path.iterdir() if item.is_file())
        plan = LoadPlan(source=str(path), source_type="folder", files=files)
        config = self._read_json(path / "config.json")
        tokenizer_config = self._read_json(path / "tokenizer_config.json")
        generation_config = self._read_json(path / "generation_config.json")
        adapter_config = self._read_json(path / "adapter_config.json")
        trainer_state = self._read_json(path / "trainer_state.json")

        plan.metadata.update(
            {
                "name": config.get("_name_or_path") or path.name,
                "config": config,
                "tokenizer_config": tokenizer_config,
                "generation_config": generation_config,
                "adapter_config": adapter_config,
                "trainer_state": trainer_state,
            }
        )
        plan.model_type = config.get("model_type") or self._infer_model_type(path.name)
        architectures = config.get("architectures") or []
        plan.architecture = architectures[0] if architectures else None
        plan.tokenizer = "found" if self._has_any(files, ["tokenizer.json", "tokenizer_config.json", "vocab.json", "spiece.model"]) else None
        plan.weight_format = self._detect_weight_format(files)
        plan.precision = self._normalize_precision(
            config.get("torch_dtype")
            or config.get("dtype")
            or (config.get("quantization_config") or {}).get("bnb_4bit_compute_dtype")
        )
        plan.load_as = "adapter" if adapter_config or self._has_any(files, ["adapter_model.bin", "adapter_model.safetensors"]) else "full_model"
        plan.has_optimizer = self._contains_name(files, "optimizer")
        plan.has_scheduler = self._contains_name(files, "scheduler")
        plan.has_training_state = bool(trainer_state) or self._has_any(files, ["trainer_state.json", "training_args.bin"])
        self._apply_training_state(plan, trainer_state)
        return plan

    def _inspect_file(self, path: Path) -> LoadPlan:
        suffix = path.suffix.lower()
        plan = LoadPlan(
            source=str(path),
            source_type="file",
            files=[path.name],
            weight_format=self.weight_suffixes.get(suffix),
            load_as="full_model",
        )
        if suffix in {".pth", ".pt", ".ckpt", ".bin"}:
            self._inspect_torch_checkpoint(plan, path)
        return plan

    def _inspect_remote_id(self, source: str) -> LoadPlan:
        return LoadPlan(
            source=source,
            source_type="huggingface",
            model_type=self._infer_model_type(source),
            tokenizer="auto",
            weight_format="auto",
            precision="auto",
            load_as="full_model",
            metadata={"name": source},
        )

    def _inspect_torch_checkpoint(self, plan: LoadPlan, path: Path) -> None:
        try:
            checkpoint = load_torch_checkpoint(path, map_location="cpu")
        except Exception as exc:
            plan.metadata["inspection_warning"] = str(exc)
            return
        if not isinstance(checkpoint, dict):
            return
        config = dict(checkpoint.get("config") or {})
        plan.metadata["config"] = config
        plan.model_type = config.get("model_type") or self._infer_model_type(path.name)
        plan.architecture = config.get("architecture")
        plan.precision = self._normalize_precision(config.get("torch_dtype") or config.get("dtype"))
        plan.tokenizer = "found" if checkpoint.get("tokenizer_metadata") or checkpoint.get("tokenizer") else None
        plan.has_optimizer = checkpoint.get("optimizer_state_dict") is not None
        plan.has_scheduler = checkpoint.get("scheduler_state_dict") is not None
        plan.has_training_state = any(key in checkpoint for key in ("train_history", "current_epoch", "global_step"))
        self._apply_training_state(
            plan,
            {
                "epoch": checkpoint.get("current_epoch"),
                "global_step": checkpoint.get("global_step") or checkpoint.get("current_batch"),
            },
        )

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                value = json.load(f)
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            logger.warning("Could not inspect JSON metadata at %s: %s", path, exc)
            return {}

    def _detect_weight_format(self, files: List[str]) -> Optional[str]:
        for suffix, name in self.weight_suffixes.items():
            if any(file_name.endswith(suffix) for file_name in files):
                return name
        return None

    def _apply_training_state(self, plan: LoadPlan, state: Dict[str, Any]) -> None:
        if not state:
            plan.can_resume_training = plan.has_training_state and (plan.has_optimizer or plan.has_scheduler)
            return
        epoch = state.get("epoch")
        step = state.get("global_step") or state.get("step")
        plan.resume_epoch = int(epoch) if epoch is not None else None
        plan.resume_step = int(step) if step is not None else None
        plan.can_resume_training = plan.has_training_state or plan.resume_epoch is not None or plan.resume_step is not None

    def _normalize_precision(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).lower().replace("torch.", "")
        aliases = {
            "float32": "fp32",
            "float": "fp32",
            "float16": "fp16",
            "half": "fp16",
            "bfloat16": "bf16",
            "8bit": "int8",
            "4bit": "int4",
        }
        return aliases.get(text, text)

    def _infer_model_type(self, value: str) -> Optional[str]:
        text = value.lower()
        for name in ("qwen", "llama", "mistral"):
            if name in text:
                return name
        return None

    def _has_any(self, files: List[str], names: List[str]) -> bool:
        return any(name in files for name in names)

    def _contains_name(self, files: List[str], part: str) -> bool:
        return any(part in file_name.lower() for file_name in files)


_INSPECTORS: List[ModelInspector] = [DefaultModelInspector()]


def register_model_inspector(inspector: ModelInspector) -> None:
    """Register a custom model-source inspector."""

    _INSPECTORS.insert(0, inspector)


def inspect_model_source(
    source: Any,
    auto_detect: bool = True,
    inspectors: Optional[Iterable[ModelInspector]] = None,
    **overrides: Any,
) -> LoadPlan:
    """Inspect a model source and merge explicit user settings over detections."""

    active_inspectors = list(inspectors) if inspectors is not None else list(_INSPECTORS)
    if auto_detect:
        for inspector in active_inspectors:
            if inspector.can_inspect(source):
                plan = inspector.inspect(source)
                break
        else:
            plan = LoadPlan(source=str(source))
    else:
        plan = LoadPlan(source=str(source))
    plan.apply_overrides(overrides)
    logger.info("SmartLoader inspection complete for %s", source)
    return plan


class SmartLoader:
    """High-level loading entry point with inspection and manual overrides."""

    @classmethod
    def register_inspector(cls, inspector: ModelInspector) -> None:
        register_model_inspector(inspector)

    @classmethod
    def inspect(cls, source: Any, auto_detect: bool = True, **overrides: Any) -> LoadPlan:
        return inspect_model_source(source, auto_detect=auto_detect, **overrides)

    @classmethod
    def load(
        cls,
        source: Any,
        auto_detect: bool = True,
        map_location: str = "cpu",
        **overrides: Any,
    ) -> LoadedCheckpoint:
        plan = cls.inspect(source, auto_detect=auto_detect, **overrides)
        loaded = load_external_model(source, map_location=map_location)
        if plan.load_optimizer is False:
            loaded.optimizer_state_dict = None
        loaded.metadata["smart_load_plan"] = plan.to_dict()
        loaded.metadata["load_report"] = plan.format_report()
        return loaded
