"""Resource limits and runtime resource inspection."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class ResourceLimits:
    """Soft resource limits used by data workflows."""

    max_records: Optional[int] = None
    max_record_chars: Optional[int] = None
    max_bytes: Optional[int] = None

    def check_record(self, record: dict[str, Any], index: int) -> None:
        if self.max_record_chars is not None:
            size = sum(len(str(value)) for value in record.values())
            if size > self.max_record_chars:
                raise ValueError(f"Record {index} exceeds max_record_chars={self.max_record_chars}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceSelection:
    """Validated runtime device and precision selection."""

    requested_device: str
    selected_device: str
    requested_precision: str
    selected_precision: str
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceConfig:
    """Explicit CPU/CUDA device and precision request."""

    device: str = "auto"
    precision: str = "auto"
    allow_fallback: bool = False

    def resolve(self) -> DeviceSelection:
        """Validate and resolve the requested runtime device."""

        try:
            import torch
        except Exception as exc:
            if self.device not in {"auto", "cpu"}:
                raise ConfigurationError("PyTorch is required for CUDA device validation.") from exc
            return DeviceSelection(self.device, "cpu", self.precision, "float32", self.device == "auto")

        requested = self.device
        warnings: list[str] = []
        if requested == "auto":
            selected = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif requested == "cpu":
            selected = "cpu"
        elif requested == "cuda":
            if not torch.cuda.is_available():
                if not self.allow_fallback:
                    raise ConfigurationError("CUDA was requested but no CUDA runtime device is available.")
                selected = "cpu"
                warnings.append("CUDA unavailable; fell back to CPU because allow_fallback=True.")
            else:
                selected = "cuda:0"
        elif requested.startswith("cuda:"):
            if not torch.cuda.is_available():
                if not self.allow_fallback:
                    raise ConfigurationError(f"{requested} was requested but CUDA is unavailable.")
                selected = "cpu"
                warnings.append(f"{requested} unavailable; fell back to CPU because allow_fallback=True.")
            else:
                try:
                    index = int(requested.split(":", 1)[1])
                except ValueError as exc:
                    raise ConfigurationError(f"Invalid CUDA device index in {requested!r}.") from exc
                if index < 0 or index >= torch.cuda.device_count():
                    raise ConfigurationError(f"CUDA device index {index} is not available; device_count={torch.cuda.device_count()}.")
                selected = requested
        else:
            raise ConfigurationError(f"Unsupported device value: {requested!r}.")

        precision = _resolve_precision(self.precision, selected)
        return DeviceSelection(requested, selected, self.precision, precision, selected != requested and requested != "auto", warnings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_precision(precision: str, selected_device: str) -> str:
    normalized = {
        "auto": "auto",
        "float32": "float32",
        "fp32": "float32",
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
    }.get(precision)
    if normalized is None:
        raise ConfigurationError(f"Unsupported precision value: {precision!r}.")
    if normalized == "auto":
        return "float16" if selected_device.startswith("cuda") else "float32"
    if selected_device == "cpu" and normalized in {"float16", "bfloat16"}:
        raise ConfigurationError(f"{normalized} precision is not supported by ArcLM's CPU execution path.")
    return normalized


def resource_info() -> dict[str, Any]:
    """Return basic local resource information."""

    info = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count()
    except Exception:
        info["cuda_available"] = False
        info["cuda_device_count"] = 0
    return info


__all__ = ["DeviceConfig", "DeviceSelection", "ResourceLimits", "resource_info"]
