"""Streaming and repeatable dataset source abstractions."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from .exceptions import DatasetFormatError
from .exceptions import OptionalDependencyError


Record = dict[str, Any]


@dataclass(frozen=True)
class DatasetSourceMetadata:
    """Metadata describing a dataset source without exposing record content."""

    source_type: str
    path: Optional[str] = None
    format: Optional[str] = None
    streaming: bool = True
    seekable: bool = False
    one_shot: bool = False
    record_count: Optional[int] = None
    encoding: str = "utf-8"
    shards: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetSource:
    """Iterable dataset source for records or streaming files.

    File-backed sources are repeatable and avoid loading all records into
    memory. Iterator-backed sources are explicitly marked one-shot unless a
    factory is provided.
    """

    def __init__(
        self,
        iterator_factory: Callable[[], Iterator[Record]],
        *,
        metadata: DatasetSourceMetadata,
        cleanup: Optional[Callable[[], None]] = None,
    ):
        self._iterator_factory = iterator_factory
        self.metadata = metadata
        self._cleanup = cleanup
        self._closed = False
        self._iterated = False

    def __iter__(self) -> Iterator[Record]:
        if self._closed:
            raise DatasetFormatError("DatasetSource is closed.")
        if self.metadata.one_shot and self._iterated:
            raise DatasetFormatError("DatasetSource wraps a one-shot iterator and cannot be iterated twice.")
        self._iterated = True
        yield from self._iterator_factory()

    def __enter__(self) -> "DatasetSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release source resources if a custom cleanup callback was supplied."""

        if not self._closed and self._cleanup:
            self._cleanup()
        self._closed = True

    @property
    def streaming(self) -> bool:
        return self.metadata.streaming

    @property
    def seekable(self) -> bool:
        return self.metadata.seekable

    @property
    def one_shot(self) -> bool:
        return self.metadata.one_shot

    def to_list(self, limit: Optional[int] = None) -> list[Record]:
        """Materialize records, optionally limiting the count."""

        records: list[Record] = []
        for index, record in enumerate(self):
            if limit is not None and index >= limit:
                break
            records.append(record)
        return records

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "DatasetSource":
        if isinstance(records, Sequence):
            cached = [dict(row) for row in records]
            return cls(
                lambda: (dict(row) for row in cached),
                metadata=DatasetSourceMetadata(
                    source_type="memory",
                    streaming=False,
                    seekable=True,
                    one_shot=False,
                    record_count=len(cached),
                ),
            )
        iterator = iter(records)
        return cls(
            lambda: (dict(row) for row in iterator),
            metadata=DatasetSourceMetadata(source_type="iterator", streaming=True, one_shot=True),
        )

    @classmethod
    def from_iterator(cls, records: Iterable[Mapping[str, Any]], *, one_shot: bool = True) -> "DatasetSource":
        if one_shot:
            iterator = iter(records)
            factory = lambda: (dict(row) for row in iterator)
        else:
            cached = list(records)
            factory = lambda: (dict(row) for row in cached)
        return cls(
            factory,
            metadata=DatasetSourceMetadata(
                source_type="iterator",
                streaming=one_shot,
                seekable=not one_shot,
                one_shot=one_shot,
                record_count=None if one_shot else len(cached),  # type: ignore[possibly-used-before-assignment]
            ),
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        streaming: bool = True,
        encoding: str = "utf-8",
        blank_lines: str = "skip",
        malformed: str = "raise",
    ) -> "DatasetSource":
        return open_dataset(path, format="jsonl", streaming=streaming, encoding=encoding, blank_lines=blank_lines, malformed=malformed)

    @classmethod
    def from_text(
        cls,
        path: str | Path,
        *,
        streaming: bool = True,
        encoding: str = "utf-8",
        blank_lines: str = "skip",
    ) -> "DatasetSource":
        return open_dataset(path, format="txt", streaming=streaming, encoding=encoding, blank_lines=blank_lines)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        streaming: bool = True,
        encoding: str = "utf-8",
        malformed: str = "raise",
    ) -> "DatasetSource":
        return open_dataset(path, format="csv", streaming=streaming, encoding=encoding, malformed=malformed)

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        format: Optional[str] = None,
        encoding: str = "utf-8",
        malformed: str = "raise",
    ) -> "DatasetSource":
        root = Path(path)
        if not root.exists() or not root.is_dir():
            raise DatasetFormatError(f"Dataset directory not found: {root}")
        files = sorted(item for item in root.rglob("*") if item.is_file())
        if format:
            files = [item for item in files if item.suffix.lower().lstrip(".") == format.lower().lstrip(".")]
        shards = [str(item) for item in files]

        def iterator() -> Iterator[Record]:
            for file_path in files:
                yield from open_dataset(file_path, format=format, streaming=True, encoding=encoding, malformed=malformed)

        return cls(
            iterator,
            metadata=DatasetSourceMetadata(
                source_type="directory",
                path=str(root),
                format=format,
                streaming=True,
                seekable=True,
                one_shot=False,
                encoding=encoding,
                shards=shards,
            ),
        )

    @classmethod
    def from_huggingface(
        cls,
        path: str,
        *,
        split: str = "train",
        streaming: bool = True,
        **kwargs: Any,
    ) -> "DatasetSource":
        """Open a Hugging Face dataset lazily when ``datasets`` is installed."""

        try:
            from datasets import load_dataset
        except Exception as exc:
            raise OptionalDependencyError("Hugging Face datasets support requires arclm[hf].") from exc

        dataset = load_dataset(path, split=split, streaming=streaming, **kwargs)
        return cls(
            lambda: (dict(row) for row in dataset),
            metadata=DatasetSourceMetadata(
                source_type="huggingface",
                path=path,
                format="hf",
                streaming=streaming,
                seekable=not streaming,
                one_shot=False,
                record_count=None,
            ),
        )


def open_dataset(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    format: Optional[str] = None,
    streaming: bool = True,
    encoding: str = "utf-8",
    blank_lines: str = "skip",
    malformed: str = "raise",
) -> DatasetSource:
    """Open a dataset as an iterable :class:`DatasetSource`.

    ``malformed`` may be ``"raise"`` or ``"report"``. Report mode yields a
    record containing ``"_error"`` instead of silently discarding malformed
    input.
    """

    if malformed not in {"raise", "report"}:
        raise DatasetFormatError("malformed must be 'raise' or 'report'.")
    if blank_lines not in {"skip", "keep", "error"}:
        raise DatasetFormatError("blank_lines must be 'skip', 'keep', or 'error'.")
    if not isinstance(source, (str, Path)):
        return DatasetSource.from_records(source)

    path = Path(source)
    if path.is_dir():
        return DatasetSource.from_directory(path, format=format, encoding=encoding, malformed=malformed)
    if not path.exists():
        raise DatasetFormatError(f"Dataset file not found: {path}")

    fmt = (format or path.suffix.lower().lstrip(".") or "txt").lower()
    if fmt == "jsonl":
        iterator = lambda: _iter_jsonl(path, encoding, blank_lines, malformed)
    elif fmt == "json":
        iterator = lambda: _iter_json(path, encoding, malformed)
        streaming = False
    elif fmt == "txt":
        iterator = lambda: _iter_text(path, encoding, blank_lines)
    elif fmt == "csv":
        iterator = lambda: _iter_csv(path, encoding, malformed)
    else:
        raise DatasetFormatError(f"Unsupported dataset format: {fmt}")

    if not streaming:
        rows = list(iterator())
        return DatasetSource.from_records(rows)

    return DatasetSource(
        iterator,
        metadata=DatasetSourceMetadata(
            source_type="file",
            path=str(path),
            format=fmt,
            streaming=True,
            seekable=True,
            one_shot=False,
            encoding=encoding,
        ),
    )


def _iter_jsonl(path: Path, encoding: str, blank_lines: str, malformed: str) -> Iterator[Record]:
    with path.open("r", encoding=encoding) as handle:
        for index, line in enumerate(handle):
            text = line.rstrip("\n")
            if not text.strip():
                if blank_lines == "skip":
                    continue
                if blank_lines == "keep":
                    yield {"text": ""}
                    continue
                raise DatasetFormatError(f"{path}: record {index}: blank line")
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                if malformed == "report":
                    yield {"_error": str(exc), "_path": str(path), "_index": index}
                    continue
                raise DatasetFormatError(f"{path}: record {index}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                message = f"{path}: record {index}: JSONL value must be an object"
                if malformed == "report":
                    yield {"_error": message, "_path": str(path), "_index": index}
                    continue
                raise DatasetFormatError(message)
            yield dict(value)


def _iter_json(path: Path, encoding: str, malformed: str) -> Iterator[Record]:
    try:
        value = json.loads(path.read_text(encoding=encoding))
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"{path}: invalid JSON: {exc}") from exc
    if isinstance(value, dict):
        value = value.get("records", [value])
    if not isinstance(value, list):
        raise DatasetFormatError(f"{path}: JSON dataset must be an object or list")
    for index, row in enumerate(value):
        if isinstance(row, dict):
            yield dict(row)
        elif malformed == "report":
            yield {"_error": "JSON item is not an object", "_path": str(path), "_index": index}
        else:
            raise DatasetFormatError(f"{path}: record {index}: JSON item is not an object")


def _iter_text(path: Path, encoding: str, blank_lines: str) -> Iterator[Record]:
    with path.open("r", encoding=encoding) as handle:
        for index, line in enumerate(handle):
            text = line.rstrip("\n")
            if not text.strip() and blank_lines == "skip":
                continue
            if not text.strip() and blank_lines == "error":
                raise DatasetFormatError(f"{path}: record {index}: blank line")
            yield {"text": text}


def _iter_csv(path: Path, encoding: str, malformed: str) -> Iterator[Record]:
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise DatasetFormatError(f"{path}: CSV file has no header")
        for index, row in enumerate(reader):
            if row is None:
                if malformed == "report":
                    yield {"_error": "CSV row could not be parsed", "_path": str(path), "_index": index}
                    continue
                raise DatasetFormatError(f"{path}: record {index}: malformed CSV row")
            yield dict(row)


__all__ = ["DatasetSource", "DatasetSourceMetadata", "open_dataset"]
