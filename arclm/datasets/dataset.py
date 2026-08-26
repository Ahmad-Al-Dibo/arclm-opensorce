"""Public Dataset abstraction over ArcLM's shared Data Engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..tokenizers import ChatTemplate, Tokenizer
from .input_pipeline import ModelInput, ModelInputPreparer
from .sources import DataEngine
from .transforms import FieldMapTransform, FunctionTransform, Transform, clean_records, record_text, shuffled


@dataclass(frozen=True)
class DatasetInspection:
    """Structured dataset inspection report."""

    source: str
    format: str
    size_bytes: int
    records: int
    characters: int
    estimated_tokens: int
    fields: list[str] = field(default_factory=list)
    schema: dict[str, str] = field(default_factory=dict)
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    text_lengths: dict[str, float] = field(default_factory=dict)
    splits: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreparedDataset:
    """Prepared next-token training data."""

    tokenizer: Tokenizer
    encoded: list[int]
    train_loader: Any
    block_size: int
    batch_size: int
    model_inputs: ModelInput | None = None


@dataclass
class DatasetSplit:
    """Organized train/validation/test datasets."""

    train: "Dataset"
    validation: "Dataset | None" = None
    test: "Dataset | None" = None

    def to_dict(self) -> dict[str, int]:
        return {
            "train": len(self.train.records),
            "validation": len(self.validation.records) if self.validation is not None else 0,
            "test": len(self.test.records) if self.test is not None else 0,
        }

    def prepare(self, **kwargs: Any) -> PreparedDataset:
        return self.train.prepare(**kwargs)


@dataclass
class Dataset:
    """ArcLM-owned dataset handle for all API levels."""

    records: list[dict[str, Any]]
    source: str = "<memory>"
    format: str = "records"
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    inspection: DatasetInspection | None = None
    prepared: PreparedDataset | None = None
    stream_factory: Callable[[], Iterable[dict[str, Any]]] | None = None
    splits: dict[str, "Dataset"] = field(default_factory=dict)

    @classmethod
    def load(cls, source: str | Path | Iterable[dict[str, Any]] | Any, *, format: str | None = None, lazy: bool = False) -> "Dataset":
        """Load data through the shared Data Engine."""

        result = DataEngine().ingest(source, format=format, lazy=lazy)
        dataset = cls(
            records=list(result.records),
            source=result.source,
            format=result.format,
            metadata=dict(result.metadata),
            stream_factory=result.stream_factory,
        )
        dataset.inspection = dataset.inspect()
        return dataset

    def materialize(self) -> "Dataset":
        """Load lazy records into memory when a concrete operation needs them."""

        if self.stream_factory is not None and not self.records:
            self.records = [dict(record) for record in self.stream_factory()]
            self.stream_factory = None
            self.metadata["lazy_materialized"] = True
        return self

    def inspect(self) -> DatasetInspection:
        """Return a structured dataset report.

        This intentionally also works as ``Dataset.inspect(path)`` because
        Python passes the source as ``self`` when called on the class.
        """

        if not isinstance(self, Dataset):
            return Dataset.load(self).inspect()

        self.materialize()
        text = self.text()
        fields = sorted({key for record in self.records for key in record if not str(key).startswith("_")})
        schema = {field: self._field_type(field) for field in fields}
        missing = {field: sum(1 for record in self.records if self._is_missing(record.get(field))) for field in fields}
        texts = [record_text(record) for record in self.records]
        duplicates = len(texts) - len(set(texts))
        lengths = [len(item) for item in texts]
        warnings: list[str] = []
        if not text.strip():
            warnings.append("dataset_is_empty")
        if duplicates:
            warnings.append("duplicates_detected")
        if any(count for count in missing.values()):
            warnings.append("missing_values_detected")
        split_counts = {name: len(dataset.records) for name, dataset in self.splits.items()}

        return DatasetInspection(
            source=self.source,
            format=self.format,
            size_bytes=self._size_bytes(text),
            records=len(self.records),
            characters=len(text),
            estimated_tokens=len(text.split()),
            fields=fields,
            schema=schema,
            missing_values=missing,
            duplicates=duplicates,
            text_lengths={
                "min": float(min(lengths)) if lengths else 0.0,
                "max": float(max(lengths)) if lengths else 0.0,
                "avg": float(sum(lengths) / len(lengths)) if lengths else 0.0,
            },
            splits=split_counts,
            metadata=dict(self.metadata),
            warnings=warnings,
        )

    def clean(
        self,
        *,
        remove_empty: bool = True,
        remove_invalid: bool = True,
        deduplicate: bool = False,
        normalize_whitespace: bool = False,
        lowercase: bool = False,
        text_fields: tuple[str, ...] | None = None,
        character_filter: Callable[[str], bool] | None = None,
    ) -> "Dataset":
        """Return a cleaned dataset using reusable cleaning operations."""

        self.materialize()
        records, report = clean_records(
            self.records,
            remove_empty=remove_empty,
            remove_invalid=remove_invalid,
            deduplicate=deduplicate,
            normalize_space=normalize_whitespace,
            lowercase=lowercase,
            text_fields=text_fields,
            character_filter=character_filter,
        )
        return self._derive(records, operation="clean", details=report)

    def map(self, transform: Transform | Callable[[dict[str, Any]], dict[str, Any] | None]) -> "Dataset":
        """Apply a custom transformation to each record."""

        self.materialize()
        active = transform if hasattr(transform, "__call__") else FunctionTransform(transform)
        records = []
        for record in self.records:
            updated = active(dict(record))
            if updated is not None:
                records.append(dict(updated))
        return self._derive(records, operation="map", details={"records": len(records), "transform": active.__class__.__name__})

    def filter(self, predicate: Callable[[dict[str, Any]], bool]) -> "Dataset":
        """Keep records matching a predicate."""

        self.materialize()
        records = [dict(record) for record in self.records if predicate(dict(record))]
        return self._derive(records, operation="filter", details={"records": len(records)})

    def sample(self, n: int, *, seed: int | None = None) -> "Dataset":
        """Return up to n sampled records."""

        self.materialize()
        shuffled_dataset = self.shuffle(seed=seed)
        return shuffled_dataset._derive(
            shuffled_dataset.records[: max(0, int(n))],
            operation="sample",
            details={"n": int(n), "seed": seed},
        )

    def shuffle(self, *, seed: int | None = None) -> "Dataset":
        """Return a shuffled dataset."""

        self.materialize()
        return self._derive(shuffled(self.records, seed=seed), operation="shuffle", details={"seed": seed})

    def split(self, *, train: float = 0.8, validation: float = 0.2, test: float = 0.0, seed: int | None = None, shuffle: bool = True) -> DatasetSplit:
        """Split records into train/validation/test datasets."""

        self.materialize()
        records = shuffled(self.records, seed=seed) if shuffle else [dict(record) for record in self.records]
        total = len(records)
        train_count = self._count(train, total)
        validation_count = self._count(validation, total)
        if train_count + validation_count > total:
            validation_count = max(0, total - train_count)
        test_count = self._count(test, total) if test else max(0, total - train_count - validation_count)
        train_records = records[:train_count]
        validation_records = records[train_count : train_count + validation_count]
        test_records = records[train_count + validation_count : train_count + validation_count + test_count]

        split = DatasetSplit(
            train=self._derive(train_records, operation="split.train", details={"records": len(train_records)}),
            validation=self._derive(validation_records, operation="split.validation", details={"records": len(validation_records)}) if validation_records else None,
            test=self._derive(test_records, operation="split.test", details={"records": len(test_records)}) if test_records else None,
        )
        self.splits = {name: dataset for name, dataset in {"train": split.train, "validation": split.validation, "test": split.test}.items() if dataset is not None}
        self.inspection = self.inspect()
        return split

    def balance(self, *, field: str) -> "Dataset":
        """Balance records by taking the same count from each field value."""

        self.materialize()
        groups: dict[Any, list[dict[str, Any]]] = {}
        for record in self.records:
            groups.setdefault(record.get(field), []).append(dict(record))
        if not groups:
            return self._derive([], operation="balance", details={"field": field})
        target = min(len(values) for values in groups.values())
        records = [record for values in groups.values() for record in values[:target]]
        return self._derive(records, operation="balance", details={"field": field, "per_group": target})

    def field_map(self, mapping: dict[str, str]) -> "Dataset":
        """Rename fields using a mapping."""

        return self.map(FieldMapTransform(mapping))

    def prepare(
        self,
        *,
        tokenizer: Tokenizer | None = None,
        chat_template: ChatTemplate | dict[str, Any] | None = None,
        max_vocab: int = 50000,
        block_size: int = 8,
        batch_size: int = 2,
        shuffle: bool = True,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool = False,
        packing: bool = False,
        loss_masking: bool = False,
    ) -> PreparedDataset:
        """Tokenize and batch the dataset for native causal LM training."""

        self.materialize()
        active_tokenizer = tokenizer or Tokenizer(max_vocab=max_vocab)
        if chat_template is not None and isinstance(active_tokenizer.chat_template, ChatTemplate):
            active_tokenizer.chat_template = ChatTemplate.from_dict(chat_template)
        preparer = ModelInputPreparer(chat_template=active_tokenizer.chat_template if isinstance(active_tokenizer.chat_template, ChatTemplate) else None)
        model_inputs = preparer.prepare(
            self.records,
            tokenizer=active_tokenizer,
            block_size=block_size,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            packing=packing,
            loss_masking=loss_masking,
        )
        encoded = model_inputs.input_ids
        if len(encoded) <= block_size:
            raise ValueError("Dataset is too small for the requested block_size.")
        from .torch_dataset import create_dataloader

        loader = create_dataloader(encoded, block_size=block_size, batch_size=batch_size, shuffle=shuffle)
        self.prepared = PreparedDataset(
            tokenizer=active_tokenizer,
            encoded=encoded,
            train_loader=loader,
            block_size=block_size,
            batch_size=batch_size,
            model_inputs=model_inputs,
        )
        return self.prepared

    def to_training_text(self, *, chat_template: ChatTemplate | dict[str, Any] | None = None) -> str:
        """Format records as training text using ArcLM's input pipeline."""

        self.materialize()
        template = ChatTemplate.from_dict(chat_template) if chat_template is not None else ChatTemplate.default()
        return "\n".join(ModelInputPreparer(chat_template=template).format_records(self.records))

    def text(self) -> str:
        self.materialize()
        return "\n".join(record_text(record) for record in self.records)

    def _derive(self, records: list[dict[str, Any]], *, operation: str, details: dict[str, Any]) -> "Dataset":
        dataset = Dataset(
            records=[dict(record) for record in records],
            source=self.source,
            format=self.format,
            metadata=dict(self.metadata),
            history=[*self.history, {"operation": operation, "details": dict(details)}],
        )
        dataset.inspection = dataset.inspect()
        return dataset

    def _field_type(self, field: str) -> str:
        values = [record.get(field) for record in self.records if record.get(field) is not None]
        if not values:
            return "unknown"
        return type(values[0]).__name__

    def _size_bytes(self, text: str) -> int:
        path = Path(self.source)
        if self.source != "<memory>" and path.exists() and path.is_file():
            return path.stat().st_size
        return len(text.encode("utf-8"))

    @staticmethod
    def _count(value: float, total: int) -> int:
        if value <= 1:
            return int(total * value)
        return min(int(value), total)

    @staticmethod
    def _is_missing(value: Any) -> bool:
        return value is None or value == ""


__all__ = ["Dataset", "DatasetInspection", "DatasetSplit", "PreparedDataset"]
