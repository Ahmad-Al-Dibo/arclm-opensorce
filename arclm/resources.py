"""Resource limits and runtime resource inspection."""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from typing import Any, Optional


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


__all__ = ["ResourceLimits", "resource_info"]
