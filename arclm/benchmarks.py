"""Small reproducible benchmark helpers for ArcLM core paths."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ._version import __version__
from .data_quality import find_duplicates
from .data_sources import open_dataset
from .schemas import validate_records


@dataclass
class BenchmarkResult:
    name: str
    duration_seconds: float
    items: int
    items_per_second: float
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    arclm_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark(name: str, function: Callable[[], int], *, metadata: dict[str, Any] | None = None) -> BenchmarkResult:
    """Run one benchmark function returning processed item count."""

    started = time.perf_counter()
    items = int(function())
    duration = time.perf_counter() - started
    return BenchmarkResult(name, duration, items, items / duration if duration else 0.0, metadata or {})


def benchmark_jsonl_loading(path: str | Path) -> BenchmarkResult:
    return benchmark("jsonl_loading", lambda: sum(1 for _ in open_dataset(path, format="jsonl", streaming=True)), metadata={"path": str(path)})


def benchmark_validation(records: list[dict[str, Any]], *, schema: str = "text") -> BenchmarkResult:
    return benchmark("dataset_validation", lambda: validate_records(records, schema=schema).total_records, metadata={"schema": schema})


def benchmark_deduplication(records: list[dict[str, Any]], *, field: str = "text") -> BenchmarkResult:
    return benchmark("exact_deduplication", lambda: find_duplicates(records, fields=[field]).total_records, metadata={"field": field})


def write_benchmark_report(results: list[BenchmarkResult], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([result.to_dict() for result in results], indent=2, sort_keys=True), encoding="utf-8")
    return target


__all__ = ["BenchmarkResult", "benchmark", "benchmark_deduplication", "benchmark_jsonl_loading", "benchmark_validation", "write_benchmark_report"]
