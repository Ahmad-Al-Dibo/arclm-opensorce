"""Unified external and native model loading, inference, training, and export.

This module gives users an ArcLM-only public API for loading native ArcLM
checkpoints, Hugging Face causal language models, and PEFT LoRA adapters.
Heavy optional libraries are imported lazily so clear dependency errors can be
raised at the point where a feature is used.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from ._version import __version__
from .config import Config
from .deprecation import warn_deprecated
from .generator import Generator
from .inference import CheckpointTokenizer, LoadedModel, load_model
from .loaders import adapt_for_training, load_external_model
from .loaders.arclm_checkpoint import ArcLMCheckpointLoader
from .loaders.smart_loader import inspect_model_source as _legacy_inspect_model_source
from .pipeline import train_model
from .sft import SFTTrainingResult, train_sft


PathLike = Union[str, Path]
PromptLike = Union[str, Sequence[Mapping[str, str]]]

_SOURCE_TYPES = {
    "native_arclm",
    "hf_full_model",
    "hf_lora_adapter",
    "safetensors",
    "pytorch_state_dict",
    "unknown",
}
_SAVE_MODES = {
    "auto",
    "none",
    "adapter_only",
    "full_model",
    "merged_model",
    "native_arclm",
    "all",
}
_SAVE_LAYOUTS = {"model_id", "flat", "timestamped", "checkpoint"}
_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
    "special_tokens_map.json",
}
_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pth", ".pt", ".ckpt"}
_HF_WEIGHT_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
}
_ADAPTER_WEIGHT_NAMES = {"adapter_model.safetensors", "adapter_model.bin"}


@dataclass
class ModelSaveConfig:
    """Developer-controlled settings for saving or exporting loaded models."""

    save_enabled: bool = True
    save_mode: str = "auto"
    save_layout: str = "model_id"
    save_tokenizer: bool = True
    save_model_config: bool = True
    save_generation_config: bool = True
    save_training_metadata: bool = True
    save_adapter_config: bool = True
    save_processor: bool = True
    save_readme: bool = True
    save_safetensors: bool = True
    merge_lora: bool = False
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.save_mode = str(self.save_mode).lower().strip()
        self.save_layout = str(self.save_layout).lower().strip()
        if self.save_mode not in _SAVE_MODES:
            raise ValueError(
                "save_mode must be one of: "
                + ", ".join(sorted(_SAVE_MODES))
                + f". Got {self.save_mode!r}."
            )
        if self.save_layout not in _SAVE_LAYOUTS:
            raise ValueError(
                "save_layout must be one of: "
                + ", ".join(sorted(_SAVE_LAYOUTS))
                + f". Got {self.save_layout!r}."
            )


@dataclass
class GenerationConfig:
    """Text generation defaults used by :class:`ExternalLoadedModel`."""

    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None
    repetition_penalty: float = 1.0
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, Sequence[int]]] = None
    stop: List[str] = field(default_factory=list)
    return_full_text: bool = False

    @classmethod
    def from_overrides(cls, base: Optional["GenerationConfig"] = None, **overrides: Any) -> "GenerationConfig":
        """Build a generation config by applying keyword overrides."""

        values = asdict(base or cls())
        generation_fields = {item.name for item in fields(cls)}
        unknown = set(overrides) - generation_fields
        if unknown:
            raise ValueError(
                "Unknown generation option(s): " + ", ".join(sorted(unknown))
            )
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)


@dataclass
class ExternalModelConfig:
    """Configuration for loading native or external causal language models."""

    source: PathLike
    base_model: Optional[str] = None
    device: Optional[str] = None
    device_map: Optional[Any] = None
    dtype: Optional[Union[str, torch.dtype]] = "auto"
    trust_remote_code: bool = True
    tokenizer_path: Optional[PathLike] = None
    max_memory: Optional[Any] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    enable_thinking: Optional[bool] = False
    default_system_prompt: Optional[str] = None
    fallback_to_simple_prompt: bool = True
    fallback_to_cpu: bool = True
    require_tokenizer: bool = True
    prefer_best_native: bool = True
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    save_config: ModelSaveConfig = field(default_factory=ModelSaveConfig)

    def __post_init__(self) -> None:
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("Use only one quantization mode: load_in_8bit or load_in_4bit.")
        if isinstance(self.generation_config, Mapping):
            self.generation_config = GenerationConfig(**dict(self.generation_config))
        if self.save_config is None:
            self.save_config = ModelSaveConfig()
        elif isinstance(self.save_config, Mapping):
            self.save_config = ModelSaveConfig(**dict(self.save_config))


@dataclass
class ModelSourceInfo:
    """Structured inspection result for a model source."""

    source: str
    source_type: str = "unknown"
    tokenizer_files_exist: bool = False
    adapter_config_exists: bool = False
    config_json_exists: bool = False
    model_weights_exist: bool = False
    detected_checkpoint_folders: List[str] = field(default_factory=list)
    recommended_loading_strategy: str = "unsupported"
    warnings: List[str] = field(default_factory=list)
    resolved_source: Optional[str] = None
    model_type: Optional[str] = None
    weight_format: Optional[str] = None
    load_as: Optional[str] = None
    base_model_name_or_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.source_type not in _SOURCE_TYPES:
            self.warnings.append(f"Unknown source type reported: {self.source_type}.")
            self.source_type = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the inspection result."""

        return asdict(self)

    def format_report(self) -> str:
        """Return a compact human-readable inspection report."""

        lines = [
            f"Source: {self.source}",
            f"Resolved source: {self.resolved_source or self.source}",
            f"Source type: {self.source_type}",
            f"Tokenizer files: {'found' if self.tokenizer_files_exist else 'not found'}",
            f"Adapter config: {'found' if self.adapter_config_exists else 'not found'}",
            f"Config: {'found' if self.config_json_exists else 'not found'}",
            f"Weights: {'found' if self.model_weights_exist else 'not found'}",
            f"Recommended loading strategy: {self.recommended_loading_strategy}",
        ]
        if self.detected_checkpoint_folders:
            lines.append(
                "Detected checkpoints: "
                + ", ".join(Path(item).name for item in self.detected_checkpoint_folders)
            )
        if self.warnings:
            lines.append("Warnings: " + " | ".join(self.warnings))
        return "\n".join(lines)

    @property
    def report(self) -> str:
        """Backward-friendly alias for :meth:`format_report`."""

        return self.format_report()


@dataclass
class ExternalLoadedModel:
    """Loaded model wrapper with a unified inference and save API."""

    model: Any
    tokenizer: Any
    config: ExternalModelConfig
    source_info: ModelSourceInfo
    device: Any
    processor: Any = None
    _native_loaded: Optional[LoadedModel] = None
    _generation_error: Optional[str] = None
    _training_method: Optional[str] = None
    _lora_merged: bool = False

    def predict(self, prompt: PromptLike, **generation_kwargs: Any) -> str:
        """Generate a single prediction from a string prompt or chat messages."""

        return self.generate(prompt, **generation_kwargs)

    def generate(self, prompt: PromptLike, **generation_kwargs: Any) -> str:
        """Generate text with either a native ArcLM or Hugging Face model."""

        if self._generation_error:
            raise RuntimeError(self._generation_error)
        if self.source_info.source_type == "native_arclm" or self._native_loaded is not None:
            return self._generate_native(prompt, **generation_kwargs)
        return self._generate_huggingface(prompt, **generation_kwargs)

    def chat(self, messages: Sequence[Mapping[str, str]], **generation_kwargs: Any) -> str:
        """Generate a response from OpenAI-style chat messages."""

        return self.generate(messages, **generation_kwargs)

    def batch_predict(self, prompts: Sequence[PromptLike], **generation_kwargs: Any) -> List[str]:
        """Generate one prediction per prompt."""

        return [self.predict(prompt, **generation_kwargs) for prompt in prompts]

    def to(self, device: Union[str, torch.device]) -> "ExternalLoadedModel":
        """Move the underlying model to a device when the backend supports it."""

        if not hasattr(self.model, "to"):
            raise RuntimeError("This loaded model does not support .to(device).")
        self.model.to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        if self._native_loaded is not None:
            self._native_loaded.device = self.device
            self._native_loaded.generator.device = self.device
        return self

    def save(
        self,
        output_dir: PathLike,
        save_config: Optional[ModelSaveConfig] = None,
    ) -> Optional[Path]:
        """Save or export this loaded model using ArcLM save settings."""

        return save_loaded_model(self, output_dir, save_config=save_config)

    def save_pretrained(
        self,
        output_dir: PathLike,
        save_config: Optional[ModelSaveConfig] = None,
    ) -> Optional[Path]:
        """Alias for :meth:`save` for Hugging Face-style user expectations."""

        return self.save(output_dir, save_config=save_config)

    def export(
        self,
        output_dir: PathLike,
        save_config: Optional[ModelSaveConfig] = None,
    ) -> Optional[Path]:
        """Alias for :meth:`save` for explicit export workflows."""

        return self.save(output_dir, save_config=save_config)

    def _generate_native(self, prompt: PromptLike, **generation_kwargs: Any) -> str:
        if not isinstance(prompt, str):
            prompt = _messages_to_simple_text(_normalize_messages(prompt), add_generation_prompt=True)
        native = self._native_loaded
        if native is None:
            raise RuntimeError("Native ArcLM generation is unavailable for this model.")
        allowed = {"max_new_tokens", "temperature", "repetition_penalty", "top_k", "top_p"}
        native_kwargs = {key: value for key, value in generation_kwargs.items() if key in allowed}
        return str(native.predict(prompt, **native_kwargs))

    def _generate_huggingface(self, prompt: PromptLike, **generation_kwargs: Any) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "This source was loaded without a Hugging Face model/tokenizer and cannot generate."
            )

        config = GenerationConfig.from_overrides(
            self.config.generation_config,
            **generation_kwargs,
        )
        rendered, used_template = _render_prompt(
            tokenizer=self.tokenizer,
            prompt=prompt,
            default_system_prompt=self.config.default_system_prompt,
            enable_thinking=self.config.enable_thinking,
            add_generation_prompt=True,
            fallback_to_simple_prompt=self.config.fallback_to_simple_prompt,
        )

        tokenized = self.tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=not used_template,
        )
        inputs = _ensure_tensor_mapping(tokenized)
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        input_device = _model_input_device(self.model, fallback=self.device)
        inputs = {
            key: value.to(input_device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        generation_kwargs = _build_hf_generation_kwargs(config, self.tokenizer)
        input_length = int(inputs["input_ids"].shape[-1])

        try:
            with torch.no_grad():
                generated = self.model.generate(**inputs, **generation_kwargs)
        except RuntimeError as exc:
            raise _augment_runtime_generation_error(exc) from exc

        if torch.is_tensor(generated):
            output_ids = generated[0]
        else:
            output_ids = generated[0]
        if not config.return_full_text:
            output_ids = output_ids[input_length:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return _apply_stop_strings(str(text), config.stop).strip()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


def inspect_model_source(source: PathLike, **kwargs: Any) -> ModelSourceInfo:
    """Inspect a local path or Hugging Face model ID before loading.

    The top-level ArcLM API returns :class:`ModelSourceInfo`. For compatibility
    with older custom-inspector code, legacy ``inspectors=...`` calls are
    accepted and converted into the new result shape.
    """

    legacy_keys = {"auto_detect", "inspectors"}
    if legacy_keys.intersection(kwargs):
        plan = _legacy_inspect_model_source(source, **kwargs)
        return _source_info_from_legacy_plan(plan)

    normalized_source = _normalize_model_source_uri(source)
    source_path, checkpoint_folders = _resolve_latest_loadable_source(normalized_source)
    info = _inspect_resolved_source(source, source_path, normalized_source=normalized_source)
    info.detected_checkpoint_folders = checkpoint_folders
    return info


def load_any_model(source: PathLike, **kwargs: Any) -> ExternalLoadedModel:
    """Load any supported ArcLM or Hugging Face causal LM source for inference.

    Deprecated:
        Use :func:`arclm.models.load_model` for new code.
    """

    warn_deprecated("load_any_model", "arclm.models.load_model", "0.9.0")

    config = _external_config_from_kwargs(source, kwargs)
    source_info = inspect_model_source(config.source)
    resolved_source = Path(source_info.resolved_source or source_info.source)

    if source_info.source_type == "native_arclm":
        return _load_native_for_inference(resolved_source, config, source_info)
    if source_info.source_type in {"hf_full_model", "hf_lora_adapter"}:
        return _load_huggingface_for_inference(
            source_info.resolved_source or source_info.source,
            config,
            source_info,
        )
    if source_info.source_type in {"safetensors", "pytorch_state_dict"}:
        return _load_state_dict_for_inference(resolved_source, config, source_info)

    warnings = "; ".join(source_info.warnings)
    suffix = f" Details: {warnings}" if warnings else ""
    raise ValueError(
        f"Unsupported or unrecognized model source: {source}. "
        "Expected a native ArcLM .pth checkpoint, Hugging Face model folder/ID, "
        "PEFT LoRA adapter folder, .safetensors file, or PyTorch state dict."
        + suffix
    )


def load_external_for_inference(source: PathLike, **kwargs: Any) -> ExternalLoadedModel:
    """Backward-friendly alias for :func:`load_any_model`."""

    return load_any_model(source, **kwargs)


def predict_external(source: PathLike, prompt: str, **kwargs: Any) -> str:
    """Load a source and return a single prediction."""

    load_kwargs, generation_kwargs = _split_load_and_generation_kwargs(kwargs)
    model = load_any_model(source, **load_kwargs)
    return model.predict(prompt, **generation_kwargs)


def fine_tune_external_model(
    model: str,
    dataset: str,
    output_dir: str,
    method: str = "sft",
    backend: str = "huggingface",
    use_lora: bool = True,
    assistant_only_loss: bool = True,
    save_config: Optional[ModelSaveConfig] = None,
    **kwargs: Any,
) -> SFTTrainingResult:
    """Fine-tune an external model through supported ArcLM training backends."""

    normalized_method = method.lower().replace("-", "_").strip()
    if normalized_method != "sft":
        raise NotImplementedError(
            f"fine_tune_external_model(method={method!r}) is not implemented. "
            "ArcLM currently supports method='sft' for external Hugging Face models. "
            "DPO, RLHF, PPO, reward modeling, and other methods are intentionally "
            "not exposed as fake features."
        )

    effective_save_config = save_config or ModelSaveConfig()
    train_kwargs = dict(kwargs)
    train_kwargs.setdefault("save_tokenizer", effective_save_config.save_tokenizer)
    result = train_sft(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        backend=backend,
        use_lora=use_lora,
        assistant_only_loss=assistant_only_loss,
        **train_kwargs,
    )
    if effective_save_config.save_training_metadata and effective_save_config.save_enabled:
        _write_training_wrapper_metadata(result, effective_save_config)
    return result


def train_native_model(*, data: str, output: str, **kwargs: Any) -> Any:
    """Train a native ArcLM model by delegating to ``train_model(mode='pretrain')``."""

    return train_model(mode="pretrain", data=data, output=output, **kwargs)


def fine_tune_native_model(
    *,
    data: str,
    output: str,
    checkpoint: str,
    **kwargs: Any,
) -> Any:
    """Fine-tune a native ArcLM checkpoint via ``train_model``."""

    return train_model(
        mode="finetune",
        data=data,
        output=output,
        checkpoint=checkpoint,
        **kwargs,
    )


def continue_native_training(
    *,
    data: str,
    output: str,
    checkpoint: str,
    **kwargs: Any,
) -> Any:
    """Continue native ArcLM training via ``train_model``."""

    return train_model(
        mode="continue_training",
        data=data,
        output=output,
        checkpoint=checkpoint,
        **kwargs,
    )


def save_loaded_model(
    loaded_model: ExternalLoadedModel,
    output_dir: PathLike,
    save_config: Optional[ModelSaveConfig] = None,
) -> Optional[Path]:
    """Save a loaded model according to :class:`ModelSaveConfig`."""

    effective_config = save_config or loaded_model.config.save_config or ModelSaveConfig()
    if isinstance(effective_config, Mapping):
        effective_config = ModelSaveConfig(**dict(effective_config))
    if not effective_config.save_enabled or effective_config.save_mode == "none":
        return None

    save_mode = _resolve_save_mode(loaded_model, effective_config)
    target_dir = _prepare_save_directory(loaded_model, output_dir, effective_config, save_mode)

    if save_mode == "native_arclm":
        _save_native_arclm(loaded_model, target_dir, effective_config, save_mode)
    elif save_mode == "adapter_only":
        _save_hf_adapter(loaded_model, target_dir, effective_config, save_mode)
    elif save_mode == "merged_model":
        _save_hf_merged_model(loaded_model, target_dir, effective_config, save_mode)
    elif save_mode in {"full_model", "all"}:
        _save_hf_full_model(loaded_model, target_dir, effective_config, save_mode)
    else:
        raise ValueError(f"Unsupported resolved save mode: {save_mode}.")

    return target_dir


def _external_config_from_kwargs(source: PathLike, kwargs: Dict[str, Any]) -> ExternalModelConfig:
    config_fields = {item.name for item in fields(ExternalModelConfig)}
    unknown = set(kwargs) - (config_fields - {"source"})
    if unknown:
        raise ValueError("Unknown load option(s): " + ", ".join(sorted(unknown)))
    return ExternalModelConfig(source=_normalize_model_source_uri(source), **kwargs)


def _normalize_model_source_uri(source: PathLike) -> PathLike:
    """Accept friendly Hugging Face URI aliases while preserving local paths."""

    if isinstance(source, Path):
        return source
    text = str(source).strip()
    lowered = text.lower()
    for prefix in ("hf://", "huggingface://"):
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip("/")
    return text


def _split_load_and_generation_kwargs(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    load_fields = {item.name for item in fields(ExternalModelConfig)} - {"source"}
    generation_fields = {item.name for item in fields(GenerationConfig)}
    load_kwargs: Dict[str, Any] = {}
    generation_kwargs: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in load_fields:
            load_kwargs[key] = value
        elif key in generation_fields:
            generation_kwargs[key] = value
        else:
            generation_kwargs[key] = value
    return load_kwargs, generation_kwargs


def _source_info_from_legacy_plan(plan: Any) -> ModelSourceInfo:
    source_type = "unknown"
    if getattr(plan, "load_as", None) == "adapter":
        source_type = "hf_lora_adapter"
    elif getattr(plan, "source_type", None) == "huggingface":
        source_type = "hf_full_model"
    elif getattr(plan, "weight_format", None) == "safetensors":
        source_type = "safetensors"
    elif getattr(plan, "weight_format", None) in {"pth", "pt", "ckpt", "bin"}:
        source_type = "pytorch_state_dict"

    files = set(getattr(plan, "files", []) or [])
    metadata = getattr(plan, "metadata", {}) or {}
    adapter_config = metadata.get("adapter_config") or {}
    config = metadata.get("config") or {}
    return ModelSourceInfo(
        source=str(getattr(plan, "source", "")),
        source_type=source_type,
        tokenizer_files_exist=bool(getattr(plan, "tokenizer", None)),
        adapter_config_exists=bool(adapter_config) or "adapter_config.json" in files,
        config_json_exists=bool(config) or "config.json" in files,
        model_weights_exist=bool(getattr(plan, "weight_format", None)),
        recommended_loading_strategy=_recommended_strategy_for_type(source_type),
        warnings=[metadata["inspection_warning"]] if metadata.get("inspection_warning") else [],
        model_type=getattr(plan, "model_type", None),
        weight_format=getattr(plan, "weight_format", None),
        load_as=getattr(plan, "load_as", None),
        base_model_name_or_path=adapter_config.get("base_model_name_or_path"),
    )


def _resolve_latest_loadable_source(source: PathLike) -> Tuple[Path, List[str]]:
    path = Path(str(source))
    checkpoint_root = path / "checkpoints"
    checkpoint_folders: List[str] = []
    if path.is_dir() and checkpoint_root.is_dir():
        candidates = [
            item
            for item in checkpoint_root.iterdir()
            if item.is_dir() or item.suffix.lower() in _WEIGHT_SUFFIXES
        ]
        candidates = [item for item in candidates if _looks_loadable_checkpoint_candidate(item)]
        if candidates:
            checkpoint_folders = [
                str(item)
                for item in sorted(candidates, key=lambda candidate: candidate.name)
                if item.is_dir()
            ]
            selected = _select_latest_checkpoint_candidate(candidates)
            return _candidate_to_loadable_source(selected), checkpoint_folders
    return path, checkpoint_folders


def _looks_loadable_checkpoint_candidate(path: Path) -> bool:
    if path.is_file():
        return path.suffix.lower() in _WEIGHT_SUFFIXES
    names = {item.name for item in path.iterdir() if item.is_file()}
    return bool(
        names.intersection({"config.json", "adapter_config.json"})
        or names.intersection(_HF_WEIGHT_NAMES)
        or names.intersection(_ADAPTER_WEIGHT_NAMES)
        or any(item.suffix.lower() in _WEIGHT_SUFFIXES for item in path.iterdir() if item.is_file())
    )


def _select_latest_checkpoint_candidate(candidates: Sequence[Path]) -> Path:
    with_steps = []
    without_steps = []
    for candidate in candidates:
        step = _extract_step_number(candidate.name)
        if step is None:
            without_steps.append(candidate)
        else:
            with_steps.append((step, candidate))
    if with_steps:
        return max(with_steps, key=lambda item: item[0])[1]
    return max(without_steps or list(candidates), key=lambda item: item.stat().st_mtime)


def _candidate_to_loadable_source(candidate: Path) -> Path:
    if candidate.is_file():
        return candidate
    direct_files = [
        item
        for item in candidate.iterdir()
        if item.is_file() and item.suffix.lower() in {".pth", ".pt", ".ckpt"}
    ]
    if direct_files and not (candidate / "config.json").exists() and not (candidate / "adapter_config.json").exists():
        return max(direct_files, key=lambda item: item.stat().st_mtime)
    return candidate


def _extract_step_number(name: str) -> Optional[int]:
    match = re.search(r"(?:step|checkpoint)[-_]?(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def _inspect_resolved_source(
    original_source: PathLike,
    resolved_path: Path,
    normalized_source: Optional[PathLike] = None,
) -> ModelSourceInfo:
    source_str = str(original_source)
    normalized_str = str(normalized_source or original_source)
    resolved_str = str(resolved_path)

    if resolved_path.exists():
        if resolved_path.is_dir():
            return _inspect_directory_source(source_str, resolved_path)
        if resolved_path.is_file():
            return _inspect_file_source(source_str, resolved_path)
        return ModelSourceInfo(
            source=source_str,
            resolved_source=resolved_str,
            warnings=[f"Model source exists but is neither a file nor a directory: {resolved_path}."],
        )

    if _looks_like_hf_model_id(normalized_str):
        return ModelSourceInfo(
            source=source_str,
            resolved_source=normalized_str,
            source_type="hf_full_model",
            tokenizer_files_exist=True,
            config_json_exists=True,
            model_weights_exist=True,
            recommended_loading_strategy="load with transformers AutoTokenizer and AutoModelForCausalLM",
            model_type=_infer_model_type_from_name(normalized_str),
            weight_format="auto",
            load_as="full_model",
        )

    return ModelSourceInfo(
        source=source_str,
        resolved_source=resolved_str,
        source_type="unknown",
        recommended_loading_strategy="unsupported",
        warnings=[
            f"Model directory or file does not exist: {resolved_path}. "
            "If this is a Hugging Face model ID, pass it in owner/model-name form."
        ],
    )


def _inspect_directory_source(source_str: str, path: Path) -> ModelSourceInfo:
    files = [item for item in path.iterdir() if item.is_file()]
    names = {item.name for item in files}
    config = _read_json(path / "config.json")
    adapter_config = _read_json(path / "adapter_config.json")
    tokenizer_files_exist = bool(names.intersection(_TOKENIZER_FILES))
    adapter_config_exists = bool(adapter_config) or "adapter_config.json" in names
    config_json_exists = "config.json" in names
    model_weights_exist = _directory_has_weights(path)

    if adapter_config_exists:
        source_type = "hf_lora_adapter"
        strategy = "load base model with transformers, then attach adapter with PEFT"
        load_as = "adapter"
    elif config_json_exists and model_weights_exist:
        source_type = "hf_full_model"
        strategy = "load with transformers AutoTokenizer and AutoModelForCausalLM"
        load_as = "full_model"
    elif config_json_exists:
        source_type = "hf_full_model"
        strategy = "load with transformers; weights may be sharded or remote-managed"
        load_as = "full_model"
    elif model_weights_exist:
        source_type = "unknown"
        strategy = "inspect weights with SmartLoader/load_external_model"
        load_as = None
    else:
        source_type = "unknown"
        strategy = "unsupported"
        load_as = None

    warnings = []
    if source_type == "hf_full_model" and not tokenizer_files_exist:
        warnings.append(
            "No tokenizer files were found in the folder. Pass tokenizer_path or use a complete Hugging Face folder."
        )
    if source_type == "hf_lora_adapter" and not adapter_config.get("base_model_name_or_path"):
        warnings.append(
            "LoRA adapter does not declare base_model_name_or_path; pass base_model when loading."
        )

    return ModelSourceInfo(
        source=source_str,
        resolved_source=str(path),
        source_type=source_type,
        tokenizer_files_exist=tokenizer_files_exist,
        adapter_config_exists=adapter_config_exists,
        config_json_exists=config_json_exists,
        model_weights_exist=model_weights_exist,
        recommended_loading_strategy=strategy,
        warnings=warnings,
        model_type=config.get("model_type") or _infer_model_type_from_name(path.name),
        weight_format=_detect_directory_weight_format(path),
        load_as=load_as,
        base_model_name_or_path=adapter_config.get("base_model_name_or_path"),
    )


def _inspect_file_source(source_str: str, path: Path) -> ModelSourceInfo:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return ModelSourceInfo(
            source=source_str,
            resolved_source=str(path),
            source_type="safetensors",
            model_weights_exist=True,
            recommended_loading_strategy="load with safetensors and adapt to ArcLM when tokenizer metadata exists",
            weight_format="safetensors",
        )

    if suffix in {".pth", ".pt", ".ckpt"}:
        if ArcLMCheckpointLoader().can_load(path):
            return ModelSourceInfo(
                source=source_str,
                resolved_source=str(path),
                source_type="native_arclm",
                tokenizer_files_exist=True,
                config_json_exists=False,
                model_weights_exist=True,
                recommended_loading_strategy="load with arclm.load_model",
                weight_format=suffix.lstrip("."),
                load_as="native_arclm",
            )
        return ModelSourceInfo(
            source=source_str,
            resolved_source=str(path),
            source_type="pytorch_state_dict",
            model_weights_exist=True,
            recommended_loading_strategy="load with SmartLoader/load_external_model; generation requires tokenizer metadata",
            weight_format=suffix.lstrip("."),
        )

    if suffix == ".bin":
        return ModelSourceInfo(
            source=source_str,
            resolved_source=str(path),
            source_type="pytorch_state_dict",
            model_weights_exist=True,
            recommended_loading_strategy="load as a PyTorch state dict; generation requires config and tokenizer metadata",
            weight_format="bin",
        )

    return ModelSourceInfo(
        source=source_str,
        resolved_source=str(path),
        source_type="unknown",
        recommended_loading_strategy="unsupported",
        warnings=[f"Unsupported checkpoint extension: {suffix or '<none>'}."],
    )


def _directory_has_weights(path: Path) -> bool:
    return any(
        item.is_file()
        and (
            item.name in _HF_WEIGHT_NAMES
            or item.name in _ADAPTER_WEIGHT_NAMES
            or item.suffix.lower() in _WEIGHT_SUFFIXES
        )
        for item in path.iterdir()
    )


def _detect_directory_weight_format(path: Path) -> Optional[str]:
    for item in path.iterdir():
        if item.is_file() and item.suffix.lower() in _WEIGHT_SUFFIXES:
            return item.suffix.lower().lstrip(".")
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _looks_like_hf_model_id(source: str) -> bool:
    text = source.strip()
    if not text or text.startswith((".", "/", "\\")):
        return False
    if any(text.endswith(suffix) for suffix in _WEIGHT_SUFFIXES):
        return False
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+(?:[/\w.-]*)?", text) or "/" not in text)


def _infer_model_type_from_name(value: str) -> Optional[str]:
    text = value.lower()
    for name in ("qwen", "llama", "mistral", "gemma", "gpt", "falcon"):
        if name in text:
            return name
    return None


def _recommended_strategy_for_type(source_type: str) -> str:
    return {
        "native_arclm": "load with arclm.load_model",
        "hf_full_model": "load with transformers AutoTokenizer and AutoModelForCausalLM",
        "hf_lora_adapter": "load base model with transformers, then attach adapter with PEFT",
        "safetensors": "load with safetensors and adapt to ArcLM when tokenizer metadata exists",
        "pytorch_state_dict": "load with torch and adapt to ArcLM when tokenizer metadata exists",
    }.get(source_type, "unsupported")


def _load_native_for_inference(
    source: Path,
    config: ExternalModelConfig,
    source_info: ModelSourceInfo,
) -> ExternalLoadedModel:
    try:
        native_loaded = load_model(
            model_path=source,
            device=config.device,
            prefer_best=config.prefer_best_native,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load native ArcLM checkpoint {source}: {exc}. "
            "Ensure the checkpoint was saved by train_model() or Trainer.save() "
            "and includes config, tokenizer mappings, and model weights."
        ) from exc

    return ExternalLoadedModel(
        model=native_loaded.model,
        tokenizer=getattr(native_loaded.generator, "tokenizer", None),
        config=config,
        source_info=source_info,
        device=native_loaded.device,
        _native_loaded=native_loaded,
    )


def _load_huggingface_for_inference(
    source: str,
    config: ExternalModelConfig,
    source_info: ModelSourceInfo,
) -> ExternalLoadedModel:
    if source_info.source_type == "hf_lora_adapter":
        return _load_hf_lora_adapter(source, config, source_info)
    return _load_hf_full_model(source, config, source_info)


def _load_hf_full_model(
    source: str,
    config: ExternalModelConfig,
    source_info: ModelSourceInfo,
) -> ExternalLoadedModel:
    AutoTokenizer, AutoModelForCausalLM = _import_transformers()
    tokenizer_source = str(config.tokenizer_path or source)
    tokenizer = _load_tokenizer(AutoTokenizer, tokenizer_source, config)
    model = _load_causal_lm(AutoModelForCausalLM, source, config)
    _ensure_tokenizer_padding(tokenizer)
    device = _move_model_if_needed(model, config)
    return ExternalLoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        source_info=source_info,
        device=device,
    )


def _load_hf_lora_adapter(
    source: str,
    config: ExternalModelConfig,
    source_info: ModelSourceInfo,
) -> ExternalLoadedModel:
    adapter_path = Path(source)
    adapter_config = _read_json(adapter_path / "adapter_config.json") if adapter_path.exists() else {}
    base_model = (
        config.base_model
        or source_info.base_model_name_or_path
        or adapter_config.get("base_model_name_or_path")
    )
    if not base_model:
        raise ValueError(
            f"LoRA adapter {source} requires base_model='model-id-or-path'. "
            "The adapter_config.json did not declare base_model_name_or_path."
        )

    PeftModel = _import_peft()
    AutoTokenizer, AutoModelForCausalLM = _import_transformers()
    tokenizer_source = str(config.tokenizer_path or _preferred_tokenizer_source(adapter_path, base_model))
    tokenizer = _load_tokenizer(AutoTokenizer, tokenizer_source, config)
    base = _load_causal_lm(AutoModelForCausalLM, base_model, config)
    try:
        model = PeftModel.from_pretrained(base, source)
    except Exception as exc:
        raise RuntimeError(
            f"Could not attach PEFT LoRA adapter {source} to base model {base_model}: {exc}. "
            "Check that the adapter was trained for this base model and that PEFT is up to date."
        ) from exc
    _ensure_tokenizer_padding(tokenizer)
    device = _move_model_if_needed(model, config)
    config.base_model = base_model
    return ExternalLoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        source_info=source_info,
        device=device,
    )


def _load_state_dict_for_inference(
    source: Path,
    config: ExternalModelConfig,
    source_info: ModelSourceInfo,
) -> ExternalLoadedModel:
    try:
        loaded = load_external_model(source, map_location=config.device or "cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load checkpoint {source}: {exc}. "
            "Unsupported checkpoint format or missing optional loader dependency."
        ) from exc

    if not loaded.stoi or not loaded.itos:
        return ExternalLoadedModel(
            model=None,
            tokenizer=None,
            config=config,
            source_info=source_info,
            device=config.device or "cpu",
            _generation_error=(
                f"Model source {source} contains weights but cannot generate text because "
                "ArcLM tokenizer mappings (stoi/itos) or tokenizer metadata are missing. "
                "Use a checkpoint saved by train_model()/Trainer.save(), pass a full "
                "Hugging Face model folder, or provide a native checkpoint with tokenizer metadata."
            ),
        )

    try:
        target_config = Config(**dict(loaded.config or {}))
        target_config.device = config.device or target_config.device
        bundle = adapt_for_training(loaded, target_config=target_config, strict=False)
        tokenizer = CheckpointTokenizer(
            stoi=loaded.stoi,
            itos=loaded.itos,
            tokenizer_type=(loaded.tokenizer_metadata or {}).get("tokenizer_type", "word"),
        )
        generator = Generator(
            model=bundle.model,
            stoi=loaded.stoi,
            itos=loaded.itos,
            block_size=bundle.config.block_size,
            device=torch.device(bundle.config.device),
            tokenizer=tokenizer,
        )
        native_loaded = LoadedModel(
            model=bundle.model,
            generator=generator,
            config=bundle.config,
            model_path=source,
            device=torch.device(bundle.config.device),
        )
    except Exception as exc:
        return ExternalLoadedModel(
            model=None,
            tokenizer=None,
            config=config,
            source_info=source_info,
            device=config.device or "cpu",
            _generation_error=(
                f"Model source {source} was loaded but could not be adapted for generation: {exc}. "
                "Check that the checkpoint architecture is ArcLM-compatible and includes tokenizer metadata."
            ),
        )

    source_info.source_type = "native_arclm"
    return ExternalLoadedModel(
        model=bundle.model,
        tokenizer=tokenizer,
        config=config,
        source_info=source_info,
        device=native_loaded.device,
        _native_loaded=native_loaded,
    )


def _import_transformers() -> Tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Loading Hugging Face models requires transformers. "
            "Install it with: pip install 'transformers>=4.51,<6'"
        ) from exc
    return AutoTokenizer, AutoModelForCausalLM


def _import_peft() -> Any:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Loading a LoRA adapter requires PEFT. Install it with: pip install peft"
        ) from exc
    return PeftModel


def _load_tokenizer(AutoTokenizer: Any, source: str, config: ExternalModelConfig) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as exc:
        if config.require_tokenizer:
            raise RuntimeError(
                f"Could not load tokenizer from {source}: {exc}. "
                "Pass tokenizer_path=... if the tokenizer lives somewhere else."
            ) from exc
        return None
    return tokenizer


def _load_causal_lm(AutoModelForCausalLM: Any, source: str, config: ExternalModelConfig) -> Any:
    model_kwargs = _hf_model_kwargs(config)
    try:
        return AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    except TypeError:
        if "torch_dtype" in model_kwargs:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
        return AutoModelForCausalLM.from_pretrained(source, **model_kwargs)
    except RuntimeError as exc:
        raise _augment_runtime_generation_error(exc) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Hugging Face causal language model from {source}: {exc}. "
            "Check the model ID/path, trust_remote_code setting, dtype, device_map, "
            "and local checkpoint files."
        ) from exc


def _hf_model_kwargs(config: ExternalModelConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    dtype = _resolve_dtype(config.dtype)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if config.device_map is not None:
        kwargs["device_map"] = config.device_map
    if config.max_memory is not None:
        kwargs["max_memory"] = config.max_memory
    if config.load_in_8bit:
        kwargs["load_in_8bit"] = True
    if config.load_in_4bit:
        kwargs["load_in_4bit"] = True
    return kwargs


def _resolve_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[Union[str, torch.dtype]]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).lower().replace("torch.", "").strip()
    if normalized == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(
            "dtype must be one of: auto, float16/fp16, bfloat16/bf16, float32/fp32."
        )
    return mapping[normalized]


def _move_model_if_needed(model: Any, config: ExternalModelConfig) -> Any:
    if config.device_map is not None:
        return config.device or config.device_map
    device = config.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(model, "to"):
        try:
            model.to(device)
        except RuntimeError as exc:
            raise _augment_runtime_generation_error(exc) from exc
    return torch.device(device)


def _preferred_tokenizer_source(adapter_path: Path, base_model: str) -> str:
    if adapter_path.exists() and adapter_path.is_dir():
        names = {item.name for item in adapter_path.iterdir() if item.is_file()}
        if names.intersection(_TOKENIZER_FILES):
            return str(adapter_path)
    return base_model


def _ensure_tokenizer_padding(tokenizer: Any) -> None:
    if tokenizer is None:
        return
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "unk_token", None)
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if eos_id is not None:
            tokenizer.pad_token_id = eos_id
        elif unk_id is not None:
            tokenizer.pad_token_id = unk_id
        else:
            raise ValueError(
                "Tokenizer has no pad_token_id, eos_token_id, or unk_token_id. "
                "Set tokenizer.pad_token before generation."
            )


def _render_prompt(
    tokenizer: Any,
    prompt: PromptLike,
    default_system_prompt: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    add_generation_prompt: bool = True,
    fallback_to_simple_prompt: bool = True,
) -> Tuple[str, bool]:
    messages = _prompt_to_messages(prompt, default_system_prompt)
    has_template = bool(
        tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None)
    )
    if has_template:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return str(tokenizer.apply_chat_template(messages, **kwargs)), True
        except TypeError:
            kwargs.pop("enable_thinking", None)
            try:
                return str(tokenizer.apply_chat_template(messages, **kwargs)), True
            except Exception:
                if not fallback_to_simple_prompt:
                    raise
        except Exception:
            if not fallback_to_simple_prompt:
                raise

    if isinstance(prompt, str) and not default_system_prompt:
        return prompt, False
    if not fallback_to_simple_prompt and has_template:
        raise RuntimeError("Tokenizer chat template failed and fallback_to_simple_prompt=False.")
    return _messages_to_simple_text(messages, add_generation_prompt=add_generation_prompt), False


def _prompt_to_messages(prompt: PromptLike, default_system_prompt: Optional[str]) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
    else:
        messages = _normalize_messages(prompt)
    if default_system_prompt and not any(message["role"] == "system" for message in messages):
        return [{"role": "system", "content": default_system_prompt}, *messages]
    return messages


def _normalize_messages(messages: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    normalized = []
    for message in messages:
        role = str(message.get("role", "user")).lower().strip() or "user"
        content = str(message.get("content", "")).strip()
        normalized.append({"role": role, "content": content})
    return normalized


def _messages_to_simple_text(
    messages: Sequence[Mapping[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    parts = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
    if add_generation_prompt:
        parts.append("assistant:")
    return "\n".join(parts)


def _ensure_tensor_mapping(tokenized: Any) -> Dict[str, Any]:
    if isinstance(tokenized, Mapping):
        return dict(tokenized)
    if hasattr(tokenized, "data") and isinstance(tokenized.data, Mapping):
        return dict(tokenized.data)
    raise TypeError("Tokenizer output must be a mapping with input_ids.")


def _build_hf_generation_kwargs(config: GenerationConfig, tokenizer: Any) -> Dict[str, Any]:
    pad_token_id = config.pad_token_id
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "unk_token_id", None)

    eos_token_id = config.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    do_sample = config.do_sample
    if do_sample is None:
        do_sample = bool(config.temperature is not None and config.temperature > 0)
    if config.temperature == 0:
        do_sample = False

    kwargs: Dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": config.repetition_penalty,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if do_sample:
        if config.temperature is not None and config.temperature > 0:
            kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
    return kwargs


def _model_input_device(model: Any, fallback: Any = None) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, Mapping):
        for value in hf_device_map.values():
            if value not in {"cpu", "disk", "meta"}:
                return torch.device(value)
    try:
        return next(model.parameters()).device
    except Exception:
        pass
    if fallback is not None and fallback != "auto":
        try:
            return torch.device(fallback)
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_stop_strings(text: str, stop: Iterable[str]) -> str:
    stops = [item for item in stop if item]
    if not stops:
        return text
    positions = [text.find(item) for item in stops if text.find(item) >= 0]
    if not positions:
        return text
    return text[: min(positions)]


def _augment_runtime_generation_error(exc: RuntimeError) -> RuntimeError:
    message = str(exc)
    if "out of memory" in message.lower():
        message += (
            "\nCUDA out of memory while loading or generating. Try device='cpu', "
            "device_map='auto', dtype='float16' or 'bfloat16', a smaller "
            "max_new_tokens value, or load_in_4bit/load_in_8bit if bitsandbytes "
            "is installed and supported."
        )
    return RuntimeError(message)


def _resolve_save_mode(loaded_model: ExternalLoadedModel, save_config: ModelSaveConfig) -> str:
    if save_config.save_mode != "auto":
        return save_config.save_mode
    source_type = loaded_model.source_info.source_type
    if source_type == "native_arclm":
        return "native_arclm"
    if source_type == "hf_lora_adapter":
        return "merged_model" if save_config.merge_lora else "adapter_only"
    if source_type == "hf_full_model":
        return "full_model"
    raise ValueError(
        f"Cannot choose automatic save mode for source type {source_type!r}. "
        "Pass save_mode explicitly if this source can be saved."
    )


def _prepare_save_directory(
    loaded_model: ExternalLoadedModel,
    output_dir: PathLike,
    save_config: ModelSaveConfig,
    save_mode: str,
) -> Path:
    base = Path(output_dir)
    if save_config.save_layout == "flat":
        target = base
    elif save_config.save_layout == "timestamped":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = base / _safe_model_folder_name(loaded_model, save_mode) / f"run_{timestamp}"
    elif save_config.save_layout == "checkpoint":
        target = base / "checkpoints" / "step-000001"
    else:
        target = base / _safe_model_folder_name(loaded_model, save_mode)

    if target.exists() and not save_config.overwrite:
        raise FileExistsError(
            f"Output path already exists: {target}. "
            "Pass ModelSaveConfig(overwrite=True) to replace it."
        )
    if target.exists() and save_config.overwrite:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _safe_model_folder_name(loaded_model: ExternalLoadedModel, save_mode: str) -> str:
    if loaded_model.source_info.source_type == "native_arclm":
        return "native_arclm_model"
    source = loaded_model.config.base_model or loaded_model.source_info.base_model_name_or_path
    if not source:
        source = loaded_model.source_info.source
    name = _safe_name(source)
    if loaded_model.source_info.source_type == "hf_lora_adapter" and save_mode in {"adapter_only", "auto"}:
        if not name.endswith("__lora"):
            name += "__lora"
    return name


def _safe_name(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    text = text.strip("/")
    text = re.sub(r"/+", "__", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "model"


def _save_native_arclm(
    loaded_model: ExternalLoadedModel,
    target_dir: Path,
    save_config: ModelSaveConfig,
    save_mode: str,
) -> None:
    if loaded_model.model is None or not hasattr(loaded_model.model, "state_dict"):
        raise RuntimeError("Native ArcLM save requires a model with state_dict().")
    native = loaded_model._native_loaded
    arclm_config = native.config if native is not None else getattr(loaded_model.model, "config", None)
    config_dict = _object_to_dict(arclm_config)
    generator = getattr(native, "generator", None)
    stoi = getattr(generator, "stoi", None)
    itos = getattr(generator, "itos", None)
    tokenizer_metadata = _native_tokenizer_metadata(generator)
    checkpoint = {
        "model_state_dict": loaded_model.model.state_dict(),
        "config": config_dict,
        "vocab_size": config_dict.get("vocab_size"),
        "stoi": stoi,
        "itos": itos,
        "tokenizer_metadata": tokenizer_metadata,
        "block_size": config_dict.get("block_size"),
    }
    torch.save(checkpoint, target_dir / "model.pth")
    if save_config.save_model_config:
        _write_json(target_dir / "config.json", config_dict)
    if save_config.save_tokenizer:
        _write_json(target_dir / "tokenizer_metadata.json", tokenizer_metadata)
    _write_save_metadata(loaded_model, target_dir, save_config, save_mode)
    if save_config.save_readme:
        _write_reload_readme(loaded_model, target_dir, save_mode)


def _save_hf_adapter(
    loaded_model: ExternalLoadedModel,
    target_dir: Path,
    save_config: ModelSaveConfig,
    save_mode: str,
) -> None:
    if not hasattr(loaded_model.model, "save_pretrained"):
        raise RuntimeError("Adapter-only save requires a PEFT model with save_pretrained().")
    _call_save_pretrained(loaded_model.model, target_dir, save_config)
    if save_config.save_tokenizer:
        _save_tokenizer(loaded_model, target_dir)
    _save_optional_hf_configs(loaded_model, target_dir, save_config)
    _write_save_metadata(loaded_model, target_dir, save_config, save_mode)
    if save_config.save_readme:
        _write_reload_readme(loaded_model, target_dir, save_mode)


def _save_hf_merged_model(
    loaded_model: ExternalLoadedModel,
    target_dir: Path,
    save_config: ModelSaveConfig,
    save_mode: str,
) -> None:
    if not hasattr(loaded_model.model, "merge_and_unload"):
        raise RuntimeError(
            "Merged-model export requires a PEFT model that supports merge_and_unload(). "
            "Use save_mode='adapter_only' or upgrade PEFT if merging is unavailable."
        )
    merged = loaded_model.model.merge_and_unload()
    loaded_model._lora_merged = True
    if not hasattr(merged, "save_pretrained"):
        raise RuntimeError("merge_and_unload() returned an object without save_pretrained().")
    _call_save_pretrained(merged, target_dir, save_config)
    if save_config.save_tokenizer:
        _save_tokenizer(loaded_model, target_dir)
    _save_optional_hf_configs(loaded_model, target_dir, save_config, model=merged)
    _write_save_metadata(loaded_model, target_dir, save_config, save_mode)
    if save_config.save_readme:
        _write_reload_readme(loaded_model, target_dir, save_mode)


def _save_hf_full_model(
    loaded_model: ExternalLoadedModel,
    target_dir: Path,
    save_config: ModelSaveConfig,
    save_mode: str,
) -> None:
    if loaded_model.source_info.source_type == "hf_lora_adapter" and save_mode == "full_model":
        raise RuntimeError(
            "A PEFT adapter cannot be saved as a full model without merging. "
            "Use save_mode='merged_model' or save_mode='adapter_only'."
        )
    if not hasattr(loaded_model.model, "save_pretrained"):
        raise RuntimeError("Full-model save requires a model with save_pretrained().")
    _call_save_pretrained(loaded_model.model, target_dir, save_config)
    if save_config.save_tokenizer:
        _save_tokenizer(loaded_model, target_dir)
    _save_optional_hf_configs(loaded_model, target_dir, save_config)
    _write_save_metadata(loaded_model, target_dir, save_config, save_mode)
    if save_config.save_readme:
        _write_reload_readme(loaded_model, target_dir, save_mode)


def _call_save_pretrained(model: Any, target_dir: Path, save_config: ModelSaveConfig) -> None:
    kwargs = {}
    if save_config.save_safetensors:
        kwargs["safe_serialization"] = True
    try:
        model.save_pretrained(target_dir, **kwargs)
    except TypeError:
        model.save_pretrained(target_dir)


def _save_tokenizer(loaded_model: ExternalLoadedModel, target_dir: Path) -> None:
    tokenizer = loaded_model.tokenizer
    if tokenizer is None:
        raise RuntimeError("save_tokenizer=True but no tokenizer is attached to this model.")
    if not hasattr(tokenizer, "save_pretrained"):
        raise RuntimeError("Attached tokenizer does not support save_pretrained().")
    tokenizer.save_pretrained(target_dir)


def _save_optional_hf_configs(
    loaded_model: ExternalLoadedModel,
    target_dir: Path,
    save_config: ModelSaveConfig,
    model: Any = None,
) -> None:
    active_model = model or loaded_model.model
    if save_config.save_generation_config:
        generation_config = getattr(active_model, "generation_config", None)
        if generation_config is not None and hasattr(generation_config, "save_pretrained"):
            generation_config.save_pretrained(target_dir)
    if save_config.save_processor and loaded_model.processor is not None:
        processor = loaded_model.processor
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(target_dir)


def _write_save_metadata(
    loaded_model: ExternalLoadedModel,
    target_dir: Path,
    save_config: ModelSaveConfig,
    save_mode: str,
) -> None:
    if not save_config.save_training_metadata:
        return
    metadata = {
        "original_source": loaded_model.source_info.source,
        "resolved_source": loaded_model.source_info.resolved_source,
        "base_model": loaded_model.config.base_model or loaded_model.source_info.base_model_name_or_path,
        "source_type": loaded_model.source_info.source_type,
        "save_mode": save_mode,
        "save_layout": save_config.save_layout,
        "arclm_version": __version__,
        "torch_version": torch.__version__,
        "transformers_version": _package_version("transformers"),
        "peft_version": _package_version("peft"),
        "dtype": str(loaded_model.config.dtype),
        "device": str(loaded_model.config.device or loaded_model.device),
        "device_map": loaded_model.config.device_map,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_method": loaded_model._training_method,
        "lora_merged": bool(loaded_model._lora_merged),
    }
    _write_json(target_dir / "arclm_metadata.json", metadata)


def _write_reload_readme(
    loaded_model: ExternalLoadedModel,
    target_dir: Path,
    save_mode: str,
) -> None:
    base_model = loaded_model.config.base_model or loaded_model.source_info.base_model_name_or_path
    lines = [
        "# ArcLM Saved Model",
        "",
        "Reload this model with ArcLM:",
        "",
        "```python",
        "from arclm import load_any_model",
        "",
    ]
    if save_mode == "adapter_only" and base_model:
        lines.extend(
            [
                "model = load_any_model(",
                f"    {str(target_dir)!r},",
                f"    base_model={base_model!r},",
                ")",
            ]
        )
    else:
        lines.append(f"model = load_any_model({str(target_dir)!r})")
    lines.extend(
        [
            "print(model.predict('Explain this model in one sentence.'))",
            "```",
            "",
            "This folder was written by ArcLM's unified save/export API.",
        ]
    )
    (target_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_training_wrapper_metadata(result: SFTTrainingResult, save_config: ModelSaveConfig) -> None:
    path = Path(result.output_dir) / "arclm_save_config.json"
    _write_json(path, asdict(save_config))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _object_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        data = value.to_dict()
    elif is_dataclass(value):
        data = asdict(value)
    elif isinstance(value, Mapping):
        data = dict(value)
    else:
        data = {
            key: getattr(value, key)
            for key in dir(value)
            if not key.startswith("_") and not callable(getattr(value, key))
        }
    return dict(data or {})


def _native_tokenizer_metadata(generator: Any) -> Dict[str, Any]:
    tokenizer = getattr(generator, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "to_checkpoint"):
        return tokenizer.to_checkpoint()
    metadata = {}
    if tokenizer is not None:
        metadata["tokenizer_type"] = getattr(tokenizer, "tokenizer_type", "word")
    return metadata


def _package_version(package_name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(package_name)
    except Exception:
        return None


__all__ = [
    "ExternalModelConfig",
    "ExternalLoadedModel",
    "GenerationConfig",
    "ModelSaveConfig",
    "ModelSourceInfo",
    "inspect_model_source",
    "load_any_model",
    "load_external_for_inference",
    "predict_external",
    "fine_tune_external_model",
    "train_native_model",
    "fine_tune_native_model",
    "continue_native_training",
    "save_loaded_model",
]
