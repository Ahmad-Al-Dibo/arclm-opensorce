"""Local run directory and artifact management."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ._version import __version__


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunMetadata:
    """Serializable run metadata."""

    run_id: str
    name: str
    path: str
    status: str = "created"
    started_at: str = field(default_factory=_now)
    ended_at: Optional[str] = None
    arclm_version: str = __version__
    python_version: str = field(default_factory=platform.python_version)
    platform: str = field(default_factory=platform.platform)
    dependencies: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    git: dict[str, Any] = field(default_factory=dict)
    seeds: dict[str, Any] = field(default_factory=dict)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    report_type: str = "run_metadata"
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Run:
    """Local run directory with configs, reports, metrics, logs, and artifacts."""

    def __init__(self, name: str, output_dir: str | Path = "runs", *, run_id: Optional[str] = None):
        safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip()) or "run"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.name = safe_name
        self.path = Path(output_dir) / f"{timestamp}_{safe_name}_{self.run_id}"
        self.metadata = RunMetadata(
            run_id=self.run_id,
            name=name,
            path=str(self.path),
            dependencies=self._dependency_versions(),
            device=self._device_info(),
            git=self._git_info(),
        )

    def __enter__(self) -> "Run":
        self.start()
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if exc is None:
            if self.metadata.status != "failed":
                self.complete()
        else:
            self.fail(str(exc))

    def start(self) -> None:
        self._ensure_dirs()
        self.metadata.status = "running"
        self._write_metadata()

    def complete(self) -> None:
        self.metadata.status = "completed"
        self.metadata.ended_at = _now()
        self._write_metadata()

    def fail(self, message: str) -> None:
        self.metadata.status = "failed"
        self.metadata.ended_at = _now()
        self.metadata.errors.append(message)
        self._write_metadata()

    def log_config(self, config: Any, name: str = "config") -> Path:
        return self._write_json(self.path / "config" / f"{name}.json", _to_jsonable(config))

    def log_report(self, report: Any, name: str) -> Path:
        return self._write_json(self.path / "reports" / f"{name}.json", _to_jsonable(report))

    def log_metric(self, name: str, value: float, *, step: Optional[int] = None) -> None:
        row = {"name": name, "value": value, "step": step, "created_at": _now()}
        self.metadata.metrics.append(row)
        metrics_path = self.path / "metrics" / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._write_metadata()

    def warn(self, message: str) -> None:
        self.metadata.warnings.append(message)
        self._write_metadata()

    def save_artifact(self, source: str | Path, *, name: Optional[str] = None) -> Path:
        source_path = Path(source)
        target = self.path / "artifacts" / (name or source_path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source_path, target)
        else:
            shutil.copy2(source_path, target)
        self.metadata.artifacts.append(str(target))
        self._write_metadata()
        return target

    def _ensure_dirs(self) -> None:
        for name in ["config", "reports", "checkpoints", "logs", "metrics", "artifacts"]:
            (self.path / name).mkdir(parents=True, exist_ok=True)

    def _write_metadata(self) -> None:
        self._ensure_dirs()
        self._write_json(self.path / "run.json", self.metadata.to_dict())

    @staticmethod
    def _write_json(path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return path

    @staticmethod
    def _dependency_versions() -> dict[str, Any]:
        versions: dict[str, Any] = {}
        for module_name in ["torch", "transformers", "tokenizers"]:
            try:
                module = __import__(module_name)
                versions[module_name] = getattr(module, "__version__", None)
            except Exception:
                versions[module_name] = None
        return versions

    @staticmethod
    def _device_info() -> dict[str, Any]:
        try:
            import torch

            return {"cuda_available": torch.cuda.is_available(), "cuda_device_count": torch.cuda.device_count()}
        except Exception:
            return {"cuda_available": False, "cuda_device_count": 0}

    @staticmethod
    def _git_info() -> dict[str, Any]:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
            return {"commit": commit, "dirty": dirty}
        except Exception:
            return {"commit": None, "dirty": None}


def list_runs(output_dir: str | Path = "runs") -> list[dict[str, Any]]:
    """List local run metadata files."""

    root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for run_json in sorted(root.glob("*/run.json")):
        try:
            rows.append(json.loads(run_json.read_text(encoding="utf-8")))
        except Exception:
            rows.append({"path": str(run_json), "status": "unreadable"})
    return rows


def inspect_run(path: str | Path) -> dict[str, Any]:
    """Read one run metadata file."""

    target = Path(path)
    if target.is_dir():
        target = target / "run.json"
    return json.loads(target.read_text(encoding="utf-8"))


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


__all__ = ["Run", "RunMetadata", "inspect_run", "list_runs"]
