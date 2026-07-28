"""Local environment diagnostics for ArcLM release-candidate checks."""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ._version import __version__
from .checkpoints import inspect_checkpoint
from .config import validate_arclm_config


@dataclass
class DoctorCheck:
    """One diagnostic check result."""

    name: str
    status: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DoctorReport:
    """Structured doctor command output."""

    checks: list[DoctorCheck]
    arclm_version: str = __version__
    report_type: str = "doctor_report"
    schema_version: str = "1.0"

    @property
    def is_valid(self) -> bool:
        return not any(check.status == "error" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arclm_version": self.arclm_version,
            "report_type": self.report_type,
            "schema_version": self.schema_version,
            "is_valid": self.is_valid,
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def run_doctor(
    *,
    config: str | Path | None = None,
    checkpoint: str | Path | None = None,
    run_dir: str | Path = "runs",
    cache_dir: str | Path = ".arclm/cache",
) -> DoctorReport:
    """Run local diagnostics without downloading models."""

    checks: list[DoctorCheck] = []
    checks.append(_python_check())
    checks.append(_dependency_check("torch"))
    checks.append(_dependency_check("transformers"))
    checks.append(_dependency_check("safetensors", optional=True))
    checks.append(_dependency_check("accelerate", optional=True))
    checks.append(_torch_runtime_check())
    checks.append(_writable_check("temporary_directory", Path(tempfile.gettempdir())))
    checks.append(_writable_check("run_directory", Path(run_dir)))
    checks.append(_writable_check("cache_directory", Path(cache_dir)))
    if config is not None:
        checks.append(_config_check(config))
    if checkpoint is not None:
        checks.append(_checkpoint_check(checkpoint))
    return DoctorReport(checks=checks)


def _python_check() -> DoctorCheck:
    version = platform.python_version()
    ok = (3, 9) <= tuple(int(part) for part in version.split(".")[:2]) < (3, 13)
    return DoctorCheck(
        "python",
        "ok" if ok else "error",
        "Supported Python version." if ok else "ArcLM supports Python >=3.9,<3.13.",
        {"version": version, "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
    )


def _dependency_check(name: str, *, optional: bool = False) -> DoctorCheck:
    try:
        module = __import__(name)
        return DoctorCheck(name, "ok", "Installed.", {"version": getattr(module, "__version__", None)})
    except Exception as exc:
        return DoctorCheck(name, "warning" if optional else "error", "Optional dependency missing." if optional else "Required dependency missing.", {"error": str(exc)})


def _torch_runtime_check() -> DoctorCheck:
    try:
        import torch
    except Exception as exc:
        return DoctorCheck("torch_runtime", "error", "PyTorch is not importable.", {"error": str(exc)})
    details: dict[str, Any] = {
        "torch": getattr(torch, "__version__", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        details["cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return DoctorCheck("torch_runtime", "ok", "Runtime inspected.", details)


def _writable_check(name: str, path: Path) -> DoctorCheck:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, delete=True) as handle:
            handle.write(b"ok")
        usage = shutil.disk_usage(path)
        return DoctorCheck(name, "ok", "Writable.", {"path": str(path), "free_bytes": usage.free})
    except Exception as exc:
        return DoctorCheck(name, "error", "Path is not writable.", {"path": str(path), "error": str(exc)})


def _config_check(path: str | Path) -> DoctorCheck:
    try:
        cfg = validate_arclm_config(path, permissive=False)
        return DoctorCheck("configuration", "ok", "Configuration is valid.", {"schema_version": cfg.schema_version})
    except Exception as exc:
        return DoctorCheck("configuration", "error", "Configuration is invalid.", {"error": str(exc)})


def _checkpoint_check(path: str | Path) -> DoctorCheck:
    report = inspect_checkpoint(path, trust="safe")
    return DoctorCheck("checkpoint", "ok" if report.is_verified else "error", "Checkpoint inspected.", report.to_dict())


__all__ = ["DoctorCheck", "DoctorReport", "run_doctor"]
