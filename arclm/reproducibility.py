"""Stable fingerprints for ArcLM datasets, configs, and workflows."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ._version import __version__


FINGERPRINT_ALGORITHM = "sha256"
FINGERPRINT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FingerprintReport:
    """Structured fingerprint with generation metadata."""

    value: str
    algorithm: str = FINGERPRINT_ALGORITHM
    schema_version: str = FINGERPRINT_SCHEMA_VERSION
    arclm_version: str = __version__
    mode: str = "content"
    reproducible: bool = True
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fingerprint(value: Any, *, mode: str = "content", max_records: Optional[int] = None) -> FingerprintReport:
    """Create a stable fingerprint for serializable values and ArcLM objects."""

    warnings: list[str] = []
    normalized = _normalize(value, mode=mode, max_records=max_records, warnings=warnings)
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    digest = hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()
    return FingerprintReport(value=digest, mode=mode, reproducible=not warnings, warnings=warnings)


def _normalize(value: Any, *, mode: str, max_records: Optional[int], warnings: list[str]) -> Any:
    if isinstance(value, FingerprintReport):
        return value.to_dict()
    if isinstance(value, Path) or isinstance(value, str):
        path = Path(value)
        if path.exists():
            return _fingerprint_path(path, mode)
        return str(value)
    if hasattr(value, "metadata") and hasattr(value.metadata, "to_dict"):
        rows: list[Any] = []
        for index, row in enumerate(value):
            if max_records is not None and index >= max_records:
                break
            rows.append(_normalize(row, mode=mode, max_records=max_records, warnings=warnings))
        return {"type": type(value).__name__, "metadata": value.metadata.to_dict(), "records": rows}
    if hasattr(value, "to_config"):
        return {"type": type(value).__name__, "config": value.to_config()}
    if hasattr(value, "to_dict"):
        return {"type": type(value).__name__, "data": value.to_dict()}
    if is_dataclass(value):
        return {"type": type(value).__name__, "data": asdict(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _redact(_normalize(val, mode=mode, max_records=max_records, warnings=warnings))
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item, mode=mode, max_records=max_records, warnings=warnings) for item in value]
    if isinstance(value, set):
        return sorted(_normalize(item, mode=mode, max_records=max_records, warnings=warnings) for item in value)
    if callable(value):
        name = getattr(value, "__qualname__", repr(value))
        module = getattr(value, "__module__", "")
        warnings.append(f"Callable {module}.{name} is not fully reproducible unless externally versioned.")
        return {"callable": f"{module}.{name}", "source_available": bool(_callable_source(value))}
    return value


def _fingerprint_path(path: Path, mode: str) -> dict[str, Any]:
    stat = path.stat()
    data: dict[str, Any] = {"path": str(path), "is_file": path.is_file(), "is_dir": path.is_dir(), "mode": mode}
    if mode == "metadata":
        data.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    elif mode == "sampled":
        data.update({"size": stat.st_size, "sample_hash": _hash_file(path, sample=True) if path.is_file() else _hash_directory(path, mode)})
    elif mode == "content":
        data.update({"content_hash": _hash_file(path) if path.is_file() else _hash_directory(path, mode)})
    else:
        raise ValueError("Fingerprint mode must be 'metadata', 'sampled', or 'content'.")
    return data


def _hash_file(path: Path, *, sample: bool = False) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if sample:
            digest.update(handle.read(8192))
            if path.stat().st_size > 8192:
                handle.seek(max(0, path.stat().st_size - 8192))
                digest.update(handle.read(8192))
        else:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _hash_directory(path: Path, mode: str) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(json.dumps(_fingerprint_path(item, mode), sort_keys=True).encode())
    return digest.hexdigest()


def _redact(value: Any) -> Any:
    if isinstance(value, str) and _looks_secret(value):
        return "[REDACTED]"
    return value


def _looks_secret(value: str) -> bool:
    lower = value.lower()
    return any(token in lower for token in ["hf_", "token=", "api_key", "secret", "password"])


def _callable_source(value: Any) -> Optional[str]:
    try:
        return inspect.getsource(value)
    except Exception:
        return None


__all__ = ["FINGERPRINT_ALGORITHM", "FINGERPRINT_SCHEMA_VERSION", "FingerprintReport", "fingerprint"]
