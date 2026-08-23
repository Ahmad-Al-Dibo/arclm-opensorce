"""Public Dataset abstraction for ArcLM's unified API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..dataset import create_dataloader
from ..tokenizer import Tokenizer


@dataclass(frozen=True)
class DatasetInspection:
    """Dataset inspection report used before planning/training."""

    source: str
    format: str
    size_bytes: int
    records: int
    characters: int
    estimated_tokens: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe inspection report."""

        return asdict(self)


@dataclass
class PreparedDataset:
    """Prepared next-token training data."""

    tokenizer: Tokenizer
    encoded: list[int]
    train_loader: Any
    block_size: int
    batch_size: int


@dataclass
class Dataset:
    """ArcLM-owned dataset handle for high-level and Lab workflows."""

    records: list[dict[str, Any]]
    source: str = "<memory>"
    format: str = "records"
    inspection: DatasetInspection | None = None
    prepared: PreparedDataset | None = None

    @classmethod
    def load(cls, source: str | Path | Iterable[dict[str, Any]], *, format: str | None = None) -> "Dataset":
        """Load a local dataset source."""

        if isinstance(source, (str, Path)):
            path = Path(source)
            detected = (format or path.suffix.lstrip(".") or "txt").lower()
            records = cls._load_path(path, detected)
            dataset = cls(records=records, source=str(path), format=detected)
        else:
            records = [dict(item) for item in source]
            dataset = cls(records=records, format=format or "records")
        dataset.inspection = dataset.inspect()
        return dataset

    def inspect(self) -> DatasetInspection:
        """Return a lightweight dataset report.

        This method intentionally also works as ``Dataset.inspect(path)`` for
        the high-level API. When called on the class, Python passes the source
        as ``self``.
        """

        if not isinstance(self, Dataset):
            return Dataset.load(self).inspect()

        text = self.text()
        size_bytes = Path(self.source).stat().st_size if self.source != "<memory>" and Path(self.source).exists() else len(text.encode("utf-8"))
        warnings: list[str] = []
        if not text.strip():
            warnings.append("dataset_is_empty")
        if len(self.records) == 1 and self.format == "txt":
            records = max(1, len([line for line in text.splitlines() if line.strip()]))
        else:
            records = len(self.records)
        return DatasetInspection(
            source=self.source,
            format=self.format,
            size_bytes=size_bytes,
            records=records,
            characters=len(text),
            estimated_tokens=len(text.split()),
            warnings=warnings,
        )

    def prepare(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        max_vocab: int = 50000,
        block_size: int = 8,
        batch_size: int = 2,
        shuffle: bool = True,
    ) -> PreparedDataset:
        """Tokenize and batch the dataset for native causal LM training."""

        active_tokenizer = tokenizer or Tokenizer(max_vocab=max_vocab)
        text = self.text()
        if active_tokenizer.stoi is None:
            active_tokenizer.build(text)
        encoded = active_tokenizer.encode_text(text)
        if len(encoded) <= block_size:
            raise ValueError("Dataset is too small for the requested block_size.")
        loader = create_dataloader(encoded, block_size=block_size, batch_size=batch_size, shuffle=shuffle)
        self.prepared = PreparedDataset(
            tokenizer=active_tokenizer,
            encoded=encoded,
            train_loader=loader,
            block_size=block_size,
            batch_size=batch_size,
        )
        return self.prepared

    def text(self) -> str:
        """Return records as training text."""

        parts = []
        for record in self.records:
            if "text" in record:
                parts.append(str(record["text"]))
            elif "prompt" in record and "completion" in record:
                parts.append(f"{record['prompt']} {record['completion']}")
            elif "instruction" in record and "output" in record:
                parts.append(f"{record['instruction']} {record['output']}")
            else:
                parts.append(" ".join(str(value) for value in record.values()))
        return "\n".join(parts)

    @staticmethod
    def _load_path(path: Path, format: str) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        if format == "txt":
            return [{"text": path.read_text(encoding="utf-8")}]
        if format == "jsonl":
            records = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
            return records
        if format == "json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [dict(item) for item in data]
            return [dict(data)]
        raise ValueError("Dataset.load() currently supports txt, jsonl, and json.")
