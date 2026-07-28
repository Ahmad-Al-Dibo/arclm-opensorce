"""Consolidated model support inspection and loading facade."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ..config_validation import normalize_device, normalize_precision
from ..exceptions import ModelCompatibilityError, ModelLoadError, OptionalDependencyError, UnsupportedModelError
from ..external_inference import GenerationConfig, ModelSourceInfo, inspect_model_source
from ..inference import load_model as load_native_model
from ..supported_models import (
    COMPATIBLE_UNTESTED,
    EXPERIMENTAL,
    NOT_SUPPORTED,
    OFFICIAL,
    get_model_capability,
)


CAUSAL_ARCHITECTURE_HINTS = (
    "CausalLM",
    "LMHeadModel",
    "GPT2LMHeadModel",
)


@dataclass
class ModelSupportReport:
    """Runtime model-support inspection result."""

    source: str
    task: str = "causal-lm"
    detected_architecture: Optional[str] = None
    model_type: Optional[str] = None
    causal_lm_compatible: bool = False
    tokenizer_available: bool = False
    training_support: str = "not verified"
    inference_support: str = "not verified"
    required_optional_dependencies: List[str] = field(default_factory=list)
    device_compatibility: str = "not checked"
    precision_compatibility: str = "not checked"
    trust_remote_code_required: bool = False
    known_limitations: List[str] = field(default_factory=list)
    support_level: str = NOT_SUPPORTED
    source_info: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        """Return whether ArcLM can attempt the requested workflow."""

        return self.support_level != NOT_SUPPORTED and self.causal_lm_compatible

    def summary(self) -> str:
        """Return a compact human-readable summary."""

        return (
            f"source={self.source} task={self.task} support={self.support_level} "
            f"architecture={self.detected_architecture or 'unknown'} "
            f"tokenizer={'yes' if self.tokenizer_available else 'no'}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    def raise_for_unsupported(self) -> None:
        """Raise when the inspected source is unsupported."""

        if not self.is_supported:
            details = "; ".join(self.errors or self.warnings or self.known_limitations)
            suffix = f" Details: {details}" if details else ""
            raise UnsupportedModelError(
                f"Model source {self.source!r} is not supported for {self.task}.{suffix}"
            )


@dataclass
class ModelBundle:
    """Loaded model bundle returned by :func:`load_model`."""

    model: Any
    tokenizer: Any
    config: Any
    device: torch.device
    precision: str
    capability_report: ModelSupportReport
    source_info: ModelSourceInfo
    source: str
    backend: str

    def predict(self, prompt: str, **generation_kwargs: Any) -> str:
        """Generate text from the loaded causal LM."""

        if self.backend == "native_arclm":
            return str(self.model.predict(prompt, **generation_kwargs))

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in dict(inputs).items()
        }
        defaults = GenerationConfig.from_overrides(**generation_kwargs)
        generation_args = {
            "max_new_tokens": defaults.max_new_tokens,
            "do_sample": defaults.do_sample if defaults.do_sample is not None else defaults.temperature > 0,
            "repetition_penalty": defaults.repetition_penalty,
        }
        if generation_args["do_sample"] and defaults.temperature is not None:
            generation_args["temperature"] = defaults.temperature
        if generation_args["do_sample"] and defaults.top_p is not None:
            generation_args["top_p"] = defaults.top_p
        pad_token_id = defaults.pad_token_id or getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_args["pad_token_id"] = pad_token_id
        eos_token_id = defaults.eos_token_id or getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generation_args["eos_token_id"] = eos_token_id
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_args)
        output_ids = output[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def save(self, output_dir: str | Path) -> Path:
        """Save this bundle to a directory."""

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        if self.backend == "native_arclm":
            raise ModelLoadError("Native ArcLM ModelBundle saving is not implemented in the facade; use Trainer.save/train_model checkpoints.")
        self.model.save_pretrained(target)
        self.tokenizer.save_pretrained(target)
        return target


def inspect_model_support(
    source: str | Path,
    *,
    task: str = "causal-lm",
    device: str = "auto",
    precision: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_path: str | Path | None = None,
) -> ModelSupportReport:
    """Inspect whether a model source can be used by ArcLM for causal LM workflows."""

    if task != "causal-lm":
        return ModelSupportReport(
            source=str(source),
            task=task,
            support_level=NOT_SUPPORTED,
            errors=["ArcLM currently supports task='causal-lm' only."],
        )

    normalized_device = normalize_device(device)
    normalized_precision = normalize_precision(precision)
    source_info = inspect_model_source(source)
    report = ModelSupportReport(
        source=str(source),
        task=task,
        required_optional_dependencies=[],
        device_compatibility=f"{normalized_device} accepted",
        precision_compatibility=f"{normalized_precision} accepted",
        source_info=source_info.to_dict(),
    )

    if source_info.source_type == "native_arclm":
        capability = get_model_capability("ArcLM native")
        report.detected_architecture = "ArcLM"
        report.model_type = "arclm"
        report.causal_lm_compatible = True
        report.tokenizer_available = True
        report.training_support = capability.training
        report.inference_support = capability.inference
        report.known_limitations = list(capability.known_limitations)
        report.support_level = OFFICIAL
        return report

    if source_info.source_type not in {"hf_full_model", "hf_lora_adapter"}:
        report.support_level = NOT_SUPPORTED
        report.errors.append(f"Unsupported source type: {source_info.source_type}.")
        return report

    report.required_optional_dependencies = ["transformers"]
    if source_info.source_type == "hf_lora_adapter":
        report.required_optional_dependencies.append("peft")

    try:
        AutoConfig, AutoTokenizer, _ = _import_transformers()
    except OptionalDependencyError as exc:
        report.errors.append(str(exc))
        report.support_level = NOT_SUPPORTED
        return report

    resolved = str(source_info.resolved_source or source_info.source)
    try:
        config = AutoConfig.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    except Exception as exc:
        report.errors.append(f"Could not load model configuration safely: {exc}")
        report.support_level = NOT_SUPPORTED
        return report

    architectures = list(getattr(config, "architectures", None) or [])
    report.detected_architecture = architectures[0] if architectures else type(config).__name__
    report.model_type = getattr(config, "model_type", source_info.model_type)
    report.causal_lm_compatible = _looks_causal_lm(config, report.detected_architecture)
    report.trust_remote_code_required = bool(getattr(config, "auto_map", None)) and not report.causal_lm_compatible

    try:
        AutoTokenizer.from_pretrained(str(tokenizer_path or resolved), trust_remote_code=trust_remote_code)
        report.tokenizer_available = True
    except Exception as exc:
        report.tokenizer_available = False
        report.errors.append(f"Could not load tokenizer: {exc}")

    if not report.causal_lm_compatible:
        report.support_level = NOT_SUPPORTED
        report.errors.append("Configuration does not look like a decoder-only causal LM.")
        return report

    if _is_gpt2_family(report.model_type, report.detected_architecture):
        capability = get_model_capability("GPT-2")
        report.support_level = OFFICIAL
        report.training_support = capability.training
        report.inference_support = capability.inference
        report.known_limitations = list(capability.known_limitations)
    elif _is_qwen_family(report.model_type, report.detected_architecture, str(source)):
        capability = get_model_capability("Qwen")
        report.support_level = EXPERIMENTAL
        report.training_support = capability.training
        report.inference_support = capability.inference
        report.known_limitations = list(capability.known_limitations)
    else:
        capability = get_model_capability("Llama")
        report.support_level = COMPATIBLE_UNTESTED
        report.training_support = capability.training
        report.inference_support = capability.inference
        report.known_limitations = list(capability.known_limitations)
    return report


def load_model(
    source: str | Path,
    *,
    task: str = "causal-lm",
    device: str = "auto",
    precision: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_path: str | Path | None = None,
    local_files_only: bool = False,
) -> ModelBundle:
    """Load a native ArcLM checkpoint or Hugging Face causal LM source."""

    support = inspect_model_support(
        source,
        task=task,
        device=device,
        precision=precision,
        trust_remote_code=trust_remote_code,
        tokenizer_path=tokenizer_path,
    )
    support.raise_for_unsupported()
    source_info = inspect_model_source(source)
    normalized_device = torch.device(normalize_device(device))
    normalized_precision = normalize_precision(precision)

    if source_info.source_type == "native_arclm":
        native = load_native_model(source_info.resolved_source or source, device=str(normalized_device))
        return ModelBundle(
            model=native,
            tokenizer=getattr(native.generator, "tokenizer", None),
            config=native.config,
            device=native.device,
            precision=normalized_precision,
            capability_report=support,
            source_info=source_info,
            source=str(source),
            backend="native_arclm",
        )

    try:
        _, AutoTokenizer, AutoModelForCausalLM = _import_transformers()
        resolved = str(source_info.resolved_source or source_info.source)
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path or resolved),
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "unk_token", None)
        kwargs = {
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        dtype = _torch_dtype(normalized_precision)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(resolved, **kwargs)
        model.to(normalized_device)
        model.eval()
    except OptionalDependencyError:
        raise
    except Exception as exc:
        raise ModelLoadError(f"Could not load causal LM source {source!r}: {exc}") from exc

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=getattr(model, "config", None),
        device=normalized_device,
        precision=normalized_precision,
        capability_report=support,
        source_info=source_info,
        source=str(source),
        backend="huggingface",
    )


def _import_transformers():
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise OptionalDependencyError(
            "Transformers is required for Hugging Face model support. Install arclm[hf]."
        ) from exc
    return AutoConfig, AutoTokenizer, AutoModelForCausalLM


def _looks_causal_lm(config: Any, architecture: Optional[str]) -> bool:
    if architecture and any(hint in architecture for hint in CAUSAL_ARCHITECTURE_HINTS):
        return True
    if getattr(config, "is_encoder_decoder", False):
        return False
    model_type = str(getattr(config, "model_type", "")).lower()
    return model_type in {"gpt2", "gpt_neo", "gptj", "qwen2", "qwen3", "llama", "mistral", "gemma", "falcon"}


def _is_gpt2_family(model_type: Optional[str], architecture: Optional[str]) -> bool:
    return str(model_type).lower() == "gpt2" or str(architecture or "").lower().startswith("gpt2")


def _is_qwen_family(model_type: Optional[str], architecture: Optional[str], source: str) -> bool:
    text = " ".join([str(model_type or ""), str(architecture or ""), source]).lower()
    return "qwen" in text


def _torch_dtype(precision: str):
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(precision)


__all__ = ["ModelBundle", "ModelSupportReport", "inspect_model_support", "load_model"]
