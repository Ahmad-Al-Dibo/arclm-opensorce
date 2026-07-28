"""Safe JSON-based ArcLM cache helpers."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ._version import __version__
from .exceptions import DatasetError


@dataclass
class CacheEntryMetadata:
    """Metadata stored next to a cached tokenized dataset."""

    key: str
    created_at: str
    arclm_version: str = __version__
    complete: bool = True
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CacheStats:
    """Cache directory statistics."""

    path: str
    entries: int
    bytes: int
    keys: list[str]
    corrupted: list[str] = field(default_factory=list)
    report_type: str = "cache_stats"
    schema_version: str = "1.0"
    arclm_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entry_dir(cache_dir: str | Path, key: str) -> Path:
    if not key or any(ch in key for ch in '\\/:*?"<>|'):
        raise DatasetError("Invalid cache key.")
    return Path(cache_dir) / key


def read_cache(cache_dir: str | Path, key: str, *, read_only: bool = False) -> Optional[dict[str, Any]]:
    """Read a JSON cache entry when it exists and is complete."""

    directory = _entry_dir(cache_dir, key)
    metadata_path = directory / "metadata.json"
    data_path = directory / "data.json"
    if not metadata_path.exists() or not data_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("complete", False):
            return None
        return {"metadata": metadata, "data": json.loads(data_path.read_text(encoding="utf-8"))}
    except Exception as exc:
        if read_only:
            return None
        raise DatasetError(f"Cache entry {key!r} is corrupted: {exc}") from exc


def write_cache(cache_dir: str | Path, key: str, data: Any, *, config: Optional[dict[str, Any]] = None, read_only: bool = False) -> Path:
    """Write cache entry atomically using JSON, avoiding unsafe deserialization."""

    if read_only:
        raise DatasetError("Cache is read-only.")
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    target = _entry_dir(root, key)
    metadata = CacheEntryMetadata(
        key=key,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config or {},
    )
    with tempfile.TemporaryDirectory(dir=root) as tmp_name:
        tmp = Path(tmp_name)
        (tmp / "data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        (tmp / "metadata.json").write_text(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        tmp.rename(target)
    return target


def inspect_cache(cache_dir: str | Path) -> CacheStats:
    """Return cache statistics without loading cached tensors."""

    root = Path(cache_dir)
    if not root.exists():
        return CacheStats(path=str(root), entries=0, bytes=0, keys=[])
    keys: list[str] = []
    corrupted: list[str] = []
    total_bytes = 0
    for entry in sorted(item for item in root.iterdir() if item.is_dir()):
        keys.append(entry.name)
        for file_path in entry.rglob("*"):
            if file_path.is_file():
                total_bytes += file_path.stat().st_size
        try:
            metadata = json.loads((entry / "metadata.json").read_text(encoding="utf-8"))
            if not metadata.get("complete", False):
                corrupted.append(entry.name)
        except Exception:
            corrupted.append(entry.name)
    return CacheStats(path=str(root), entries=len(keys), bytes=total_bytes, keys=keys, corrupted=corrupted)


def clear_cache(cache_dir: str | Path, *, key: Optional[str] = None) -> CacheStats:
    """Clear a cache directory or one key, then return updated stats."""

    root = Path(cache_dir)
    if key:
        target = _entry_dir(root, key)
        if target.exists():
            shutil.rmtree(target)
    elif root.exists():
        shutil.rmtree(root)
    return inspect_cache(root)


__all__ = ["CacheEntryMetadata", "CacheStats", "clear_cache", "inspect_cache", "read_cache", "write_cache"]
