"""ArcLM-owned data source ingestion."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


class DataSource(Protocol):
    """Research extension contract for custom data sources."""

    def load(self) -> Iterable[dict[str, Any]]: ...
    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class IngestionResult:
    """Result returned by the ingestion layer."""

    records: list[dict[str, Any]]
    source: str
    format: str
    metadata: dict[str, Any] = field(default_factory=dict)
    stream_factory: Callable[[], Iterable[dict[str, Any]]] | None = None


class DataEngine:
    """Shared ingestion engine for all ArcLM API levels."""

    SUPPORTED_FILES = {"txt", "json", "jsonl", "csv", "parquet"}

    def ingest(self, source: Any, *, format: str | None = None, lazy: bool = False) -> IngestionResult:
        if hasattr(source, "load") and callable(source.load):
            records = [dict(record) for record in source.load()]
            metadata = source.metadata() if hasattr(source, "metadata") else {}
            return IngestionResult(records=records, source=metadata.get("source", source.__class__.__name__), format=metadata.get("format", format or "custom"), metadata=metadata)

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                return self._ingest_directory(path, lazy=lazy)
            detected = (format or path.suffix.lstrip(".") or "txt").lower()
            if lazy and detected in {"txt", "jsonl", "csv"}:
                return IngestionResult(
                    records=[],
                    source=str(path),
                    format=detected,
                    metadata=self._metadata(path, detected, lazy=True),
                    stream_factory=lambda: self._load_path(path, detected),
                )
            records = list(self._load_path(path, detected))
            return IngestionResult(records=records, source=str(path), format=detected, metadata=self._metadata(path, detected, lazy=False))

        if isinstance(source, dict):
            return IngestionResult(records=[dict(source)], source="<memory>", format=format or "record", metadata={"records": 1})

        records = [self._coerce_record(item) for item in source]
        return IngestionResult(records=records, source="<memory>", format=format or "records", metadata={"records": len(records)})

    def _ingest_directory(self, path: Path, *, lazy: bool) -> IngestionResult:
        files = [item for item in sorted(path.rglob("*")) if item.is_file() and item.suffix.lstrip(".").lower() in self.SUPPORTED_FILES]
        if lazy:
            def stream():
                for item in files:
                    yield from self._load_path(item, item.suffix.lstrip(".").lower() or "txt")

            return IngestionResult(records=[], source=str(path), format="directory", metadata={"files": len(files), "lazy": True}, stream_factory=stream)
        records: list[dict[str, Any]] = []
        for item in files:
            records.extend(self._load_path(item, item.suffix.lstrip(".").lower() or "txt"))
        return IngestionResult(records=records, source=str(path), format="directory", metadata={"files": len(files), "records": len(records), "lazy": False})

    def _load_path(self, path: Path, format: str) -> Iterable[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if format == "txt":
            yield {"text": path.read_text(encoding="utf-8")}
            return
        if format == "jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        record = {"value": record}
                    record.setdefault("_line", line_number)
                    yield dict(record)
            return
        if format == "json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    yield self._coerce_record(item)
            else:
                yield self._coerce_record(data)
            return
        if format == "csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                yield from (dict(row) for row in csv.DictReader(handle))
            return
        if format == "parquet":
            try:
                import pandas as pd
            except Exception as exc:  # pragma: no cover - optional dependency path
                raise RuntimeError("Parquet loading requires pandas with a parquet engine such as pyarrow.") from exc
            for record in pd.read_parquet(path).to_dict(orient="records"):
                yield dict(record)
            return
        raise ValueError("Dataset.load() supports txt, json, jsonl, csv, parquet, directories, records, and custom DataSource objects.")

    @staticmethod
    def _coerce_record(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        return {"value": item}

    @staticmethod
    def _metadata(path: Path, format: str, *, lazy: bool) -> dict[str, Any]:
        return {
            "path": str(path),
            "format": format,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "lazy": lazy,
        }


__all__ = ["DataEngine", "DataSource", "IngestionResult"]
