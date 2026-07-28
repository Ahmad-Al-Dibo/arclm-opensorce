"""Dataset sharding, splitting, duplicate detection, and quality reports."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

from ._version import __version__
from .data_sources import DatasetSource, open_dataset
from .exceptions import DatasetError, DatasetValidationError
from .schemas import validate_records


Record = dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _record_hash(record: Mapping[str, Any], *, fields: Optional[Sequence[str]] = None, normalize: bool = False, seed: int = 0) -> str:
    if fields:
        value: Any = {field: record.get(field) for field in fields}
    else:
        value = dict(record)
    if normalize:
        value = _normalize_value(value)
    payload = f"{seed}:{_stable_json(value)}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip().lower())
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(val) for key, val in sorted(value.items())}
    return value


@dataclass
class DuplicateGroup:
    """Indexes and hash for duplicate records."""

    key_hash: str
    indexes: list[int]
    count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DuplicateReport:
    """Result returned by :func:`find_duplicates`."""

    total_records: int
    duplicate_records: int
    groups: list[DuplicateGroup]
    fields: Optional[list[str]]
    normalize: bool
    created_at: str = field(default_factory=_utc_now)
    report_type: str = "duplicate_report"
    schema_version: str = "1.0"
    arclm_version: str = __version__

    @property
    def has_duplicates(self) -> bool:
        return self.duplicate_records > 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["groups"] = [group.to_dict() for group in self.groups]
        return data


@dataclass
class DatasetShard:
    """One deterministic shard."""

    index: int
    records: list[Record]

    def __iter__(self) -> Iterator[Record]:
        yield from self.records

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class ShardReport:
    """Dataset sharding report."""

    strategy: str
    num_shards: int
    total_records: int
    counts: list[int]
    empty_shards: list[int]
    seed: int = 0
    report_type: str = "shard_report"
    schema_version: str = "1.0"
    arclm_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShardResult:
    shards: list[DatasetShard]
    report: ShardReport

    def to_dict(self) -> dict[str, Any]:
        return {"shards": [{"index": shard.index, "count": len(shard)} for shard in self.shards], "report": self.report.to_dict()}


@dataclass
class SplitReport:
    """Dataset split report."""

    strategy: str
    total_records: int
    counts: dict[str, int]
    percentages: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    seed: int = 42
    report_type: str = "split_report"
    schema_version: str = "1.0"
    arclm_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SplitResult:
    splits: dict[str, list[Record]]
    report: SplitReport

    def to_dict(self) -> dict[str, Any]:
        return {"splits": {name: len(rows) for name, rows in self.splits.items()}, "report": self.report.to_dict()}


@dataclass
class DataQualityIssue:
    """One data-quality issue without raw private content."""

    category: str
    index: Optional[int]
    field: Optional[str]
    message: str
    severity: str = "warning"
    sample: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityReport:
    """Privacy-aware data-quality report."""

    total_records: int
    metrics: dict[str, Any]
    issues: list[DataQualityIssue]
    samples: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    report_type: str = "data_quality_report"
    schema_version: str = "1.0"
    arclm_version: str = __version__
    created_at: str = field(default_factory=_utc_now)

    def summary(self) -> str:
        return f"records={self.total_records} issues={len(self.issues)} warnings={len(self.warnings)} errors={len(self.errors)}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [issue.to_dict() for issue in self.issues]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)

    def to_markdown(self) -> str:
        lines = ["# Data Quality Report", "", self.summary(), "", "## Metrics"]
        for key, value in sorted(self.metrics.items()):
            lines.append(f"- `{key}`: `{value}`")
        if self.issues:
            lines.extend(["", "## Issues"])
            for issue in self.issues[:50]:
                lines.append(f"- {issue.severity}: {issue.category} record={issue.index} field={issue.field}: {issue.message}")
        return "\n".join(lines)


def shard_dataset(
    dataset: Iterable[Mapping[str, Any]],
    *,
    num_shards: int,
    strategy: str = "contiguous",
    key: Optional[str] = None,
    seed: int = 0,
) -> ShardResult:
    """Deterministically shard records without duplication or loss."""

    if num_shards <= 0:
        raise DatasetError("num_shards must be greater than zero.")
    if strategy not in {"contiguous", "round_robin", "hash"}:
        raise DatasetError("strategy must be 'contiguous', 'round_robin', or 'hash'.")
    rows = [dict(record) for record in dataset]
    shards = [DatasetShard(index=index, records=[]) for index in range(num_shards)]
    if strategy == "contiguous":
        size = math.ceil(len(rows) / num_shards) if rows else 0
        for index, row in enumerate(rows):
            shard_index = min(index // size, num_shards - 1) if size else 0
            shards[shard_index].records.append(row)
    elif strategy == "round_robin":
        for index, row in enumerate(rows):
            shards[index % num_shards].records.append(row)
    else:
        for row in rows:
            shard_key = str(row.get(key, _stable_json(row))) if key else _stable_json(row)
            digest = hashlib.sha256(f"{seed}:{shard_key}".encode("utf-8", errors="replace")).hexdigest()
            shards[int(digest, 16) % num_shards].records.append(row)
    counts = [len(shard) for shard in shards]
    return ShardResult(
        shards=shards,
        report=ShardReport(
            strategy=strategy,
            num_shards=num_shards,
            total_records=len(rows),
            counts=counts,
            empty_shards=[index for index, count in enumerate(counts) if count == 0],
            seed=seed,
        ),
    )


def split_dataset(
    dataset: Iterable[Mapping[str, Any]],
    *,
    train: float = 0.8,
    validation: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    strategy: str = "hash",
    key: Optional[str] = None,
    group_key: Optional[str] = None,
    date_field: Optional[str] = None,
    split_field: Optional[str] = None,
) -> SplitResult:
    """Create deterministic train/validation/test splits."""

    _validate_ratios(train, validation, test)
    rows = [dict(record) for record in dataset]
    splits: dict[str, list[Record]] = {"train": [], "validation": [], "test": []}
    warnings: list[str] = []
    if split_field:
        for row in rows:
            name = str(row.get(split_field, "train")).lower()
            if name in {"val", "valid"}:
                name = "validation"
            if name not in splits:
                warnings.append(f"Unknown split value {name!r}; assigning to train.")
                name = "train"
            splits[name].append(row)
    elif strategy == "chronological":
        if not date_field:
            raise DatasetError("date_field is required for chronological split.")
        ordered = sorted(rows, key=lambda row: str(row.get(date_field, "")))
        _assign_contiguous(ordered, splits, train, validation, test)
    elif group_key:
        groups: dict[str, list[Record]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(group_key, ""))].append(row)
        for group, group_rows in sorted(groups.items()):
            name = _hash_split(group, train, validation, seed)
            splits[name].extend(group_rows)
    elif strategy == "random":
        ordered = list(rows)
        random.Random(seed).shuffle(ordered)
        _assign_contiguous(ordered, splits, train, validation, test)
    elif strategy == "hash":
        for row in rows:
            value = str(row.get(key, _stable_json(row))) if key else _stable_json(row)
            splits[_hash_split(value, train, validation, seed)].append(row)
    else:
        raise DatasetError("strategy must be 'hash', 'random', or 'chronological'.")
    counts = {name: len(value) for name, value in splits.items()}
    total = len(rows)
    percentages = {name: (count / total if total else 0.0) for name, count in counts.items()}
    for name, count in counts.items():
        if total and count == 0:
            warnings.append(f"Split {name!r} is empty.")
    return SplitResult(
        splits=splits, report=SplitReport(strategy=strategy, total_records=total, counts=counts, percentages=percentages, warnings=warnings, seed=seed)
    )


def _validate_ratios(train: float, validation: float, test: float) -> None:
    values = [train, validation, test]
    if any(value < 0 for value in values):
        raise DatasetError("Split ratios must be non-negative.")
    total = sum(values)
    if total <= 0:
        raise DatasetError("At least one split ratio must be greater than zero.")
    if abs(total - 1.0) > 1e-6:
        raise DatasetError("Split ratios must sum to 1.0.")


def _assign_contiguous(rows: list[Record], splits: dict[str, list[Record]], train: float, validation: float, test: float) -> None:
    total = len(rows)
    train_count = int(total * train)
    validation_count = int(total * validation)
    splits["train"].extend(rows[:train_count])
    splits["validation"].extend(rows[train_count : train_count + validation_count])
    splits["test"].extend(rows[train_count + validation_count :])


def _hash_split(value: str, train: float, validation: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8", errors="replace")).hexdigest()
    bucket = int(digest[:12], 16) / float(0xFFFFFFFFFFFF)
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def find_duplicates(
    dataset: Iterable[Mapping[str, Any]],
    *,
    fields: Optional[Sequence[str]] = None,
    normalize: bool = False,
) -> DuplicateReport:
    """Find exact duplicate groups by whole record or selected fields."""

    buckets: dict[str, list[int]] = defaultdict(list)
    total = 0
    for index, record in enumerate(dataset):
        total += 1
        buckets[_record_hash(record, fields=fields, normalize=normalize)].append(index)
    groups = [DuplicateGroup(key_hash=key, indexes=indexes, count=len(indexes)) for key, indexes in buckets.items() if len(indexes) > 1]
    duplicate_records = sum(group.count - 1 for group in groups)
    return DuplicateReport(
        total_records=total, duplicate_records=duplicate_records, groups=groups, fields=list(fields) if fields else None, normalize=normalize
    )


def find_near_duplicates(
    dataset: Iterable[Mapping[str, Any]],
    *,
    field: str = "text",
    threshold: float = 0.9,
    normalize: bool = True,
) -> DuplicateReport:
    """Find approximate near-duplicates with token-set Jaccard similarity."""

    if not 0 < threshold <= 1:
        raise DatasetError("threshold must be in the interval (0, 1].")
    rows = [dict(row) for row in dataset]
    token_sets = [_token_set(str(row.get(field, "")), normalize=normalize) for row in rows]
    groups: list[DuplicateGroup] = []
    used: set[int] = set()
    for index, tokens in enumerate(token_sets):
        if index in used:
            continue
        indexes = [index]
        for other_index in range(index + 1, len(token_sets)):
            if other_index in used:
                continue
            other = token_sets[other_index]
            union = tokens | other
            score = len(tokens & other) / len(union) if union else 1.0
            if score >= threshold:
                indexes.append(other_index)
                used.add(other_index)
        if len(indexes) > 1:
            groups.append(DuplicateGroup(key_hash=f"near:{index}", indexes=indexes, count=len(indexes)))
    return DuplicateReport(
        total_records=len(rows),
        duplicate_records=sum(group.count - 1 for group in groups),
        groups=groups,
        fields=[field],
        normalize=normalize,
    )


def _token_set(text: str, *, normalize: bool) -> set[str]:
    if normalize:
        text = re.sub(r"\s+", " ", text.strip().lower())
    return set(text.split())


def analyze_dataset(
    dataset: Iterable[Mapping[str, Any]],
    *,
    schema: Optional[str] = None,
    checks: Optional[Sequence[str]] = None,
    text_field: str = "text",
    include_samples: bool = False,
    redact_samples: bool = True,
    max_sample_chars: int = 120,
    tokenizer: Optional[Any] = None,
) -> DataQualityReport:
    """Analyze dataset quality without exposing full records by default."""

    enabled = set(checks or ["missing", "empty", "lengths", "duplicates", "roles", "unicode", "control_chars", "field_presence"])
    rows = [dict(row) for row in dataset]
    issues: list[DataQualityIssue] = []
    warnings: list[str] = []
    lengths: list[int] = []
    token_lengths: list[int] = []
    fields = Counter()
    language_distribution = Counter()
    prompt_output_ratios: list[float] = []

    if schema:
        validation = validate_records(rows, schema=schema, strict=False)
        for issue in validation.errors:
            issues.append(DataQualityIssue(category=issue.category, index=issue.index, field=issue.field, message=issue.message, severity="error"))

    for index, row in enumerate(rows):
        fields.update(row.keys())
        text = _extract_text(row, text_field)
        lengths.append(len(text))
        if "missing" in enabled:
            for key, value in row.items():
                if value is None:
                    issues.append(DataQualityIssue("missing_field_value", index, key, "Field value is null."))
        if "empty" in enabled and not text.strip():
            issues.append(DataQualityIssue("empty_text", index, text_field, "Text content is empty."))
        if "lengths" in enabled:
            if 0 < len(text) < 3:
                issues.append(DataQualityIssue("too_short", index, text_field, "Text content is extremely short."))
            if len(text) > 100_000:
                issues.append(DataQualityIssue("too_long", index, text_field, "Text content is extremely long."))
        if "unicode" in enabled and "\ufffd" in text:
            issues.append(DataQualityIssue("broken_unicode", index, text_field, "Replacement character detected."))
        if "control_chars" in enabled and any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text):
            issues.append(DataQualityIssue("control_character", index, text_field, "Control character detected."))
        if "binary" in enabled and text.count("\x00") > 0:
            issues.append(DataQualityIssue("suspicious_binary", index, text_field, "NUL byte detected."))
        if "roles" in enabled and "messages" in row:
            _check_roles(index, row, issues)
        if "language" in enabled:
            language_distribution[_detect_simple_language(text)] += 1
        if "token_lengths" in enabled and tokenizer is not None:
            token_lengths.append(len(_encode_with_tokenizer(tokenizer, text)))
        if "prompt_imbalance" in enabled and "prompt" in row and "completion" in row:
            prompt_len = max(1, len(str(row.get("prompt", ""))))
            prompt_output_ratios.append(len(str(row.get("completion", ""))) / prompt_len)

    duplicate_report = None
    if "duplicates" in enabled:
        duplicate_report = find_duplicates(rows, fields=[text_field] if any(text_field in row for row in rows) else None, normalize=True)
        for group in duplicate_report.groups[:50]:
            issues.append(DataQualityIssue("duplicate", group.indexes[1], text_field, f"Duplicate of record {group.indexes[0]}."))

    metrics: dict[str, Any] = {
        "field_presence": dict(fields),
        "lengths": _stats(lengths),
        "duplicate_records": duplicate_report.duplicate_records if duplicate_report else 0,
    }
    if token_lengths:
        metrics["token_lengths"] = _stats(token_lengths)
    if language_distribution:
        metrics["language_distribution"] = dict(language_distribution)
    if prompt_output_ratios:
        metrics["prompt_output_ratio"] = _stats(prompt_output_ratios)

    samples: list[dict[str, Any]] = []
    if include_samples:
        for issue in issues[:10]:
            if issue.index is not None and 0 <= issue.index < len(rows):
                samples.append({"index": issue.index, "sample": _redact_sample(rows[issue.index], redact_samples, max_sample_chars)})

    return DataQualityReport(total_records=len(rows), metrics=metrics, issues=issues, samples=samples, warnings=warnings)


def check_leakage(
    train_records: Iterable[Mapping[str, Any]],
    test_records: Iterable[Mapping[str, Any]],
    *,
    fields: Optional[Sequence[str]] = None,
    normalize: bool = True,
) -> DuplicateReport:
    """Detect exact overlap between train and test records."""

    train_hashes = {_record_hash(row, fields=fields, normalize=normalize) for row in train_records}
    leaked: list[DuplicateGroup] = []
    total = 0
    for index, row in enumerate(test_records):
        total += 1
        digest = _record_hash(row, fields=fields, normalize=normalize)
        if digest in train_hashes:
            leaked.append(DuplicateGroup(key_hash=digest, indexes=[index], count=1))
    return DuplicateReport(total_records=total, duplicate_records=len(leaked), groups=leaked, fields=list(fields) if fields else None, normalize=normalize)


def _extract_text(row: Mapping[str, Any], text_field: str) -> str:
    if text_field in row:
        return str(row.get(text_field, ""))
    if "prompt" in row and "completion" in row:
        return f"{row.get('prompt', '')}\n{row.get('completion', '')}"
    if "instruction" in row and "output" in row:
        return f"{row.get('instruction', '')}\n{row.get('input', '')}\n{row.get('output', '')}"
    if "messages" in row and isinstance(row["messages"], list):
        return "\n".join(str(message.get("content", "")) for message in row["messages"] if isinstance(message, Mapping))
    return _stable_json(row)


def _check_roles(index: int, row: Mapping[str, Any], issues: list[DataQualityIssue]) -> None:
    roles = {"system", "user", "assistant", "tool"}
    previous = None
    for message_index, message in enumerate(row.get("messages", [])):
        if not isinstance(message, Mapping):
            issues.append(DataQualityIssue("invalid_message", index, "messages", f"Message {message_index} is not an object.", "error"))
            continue
        role = message.get("role")
        if role not in roles:
            issues.append(DataQualityIssue("invalid_role", index, "messages", f"Invalid role {role!r}.", "error"))
        content = str(message.get("content", ""))
        if previous == (role, content):
            issues.append(DataQualityIssue("repeated_message", index, "messages", f"Message {message_index} repeats the previous message."))
        previous = (role, content)


def _detect_simple_language(text: str) -> str:
    if not text.strip():
        return "unknown"
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06ff")
    ascii_letters = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if arabic > ascii_letters:
        return "ar"
    if ascii_letters:
        return "en"
    return "other"


def _encode_with_tokenizer(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode_text"):
        return list(tokenizer.encode_text(text))
    encoded = tokenizer(text)
    if isinstance(encoded, Mapping):
        return list(encoded.get("input_ids", []))
    return list(encoded)


def _stats(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[len(ordered) // 2],
    }


def _redact_sample(record: Mapping[str, Any], redact: bool, max_chars: int) -> str:
    text = _stable_json(record)
    text = text[:max_chars]
    if redact:
        text = re.sub(r"[\w.+-]+@[\w.-]+", "[EMAIL]", text)
        text = re.sub(r"\b\d{3,}\b", "[NUMBER]", text)
    return text


__all__ = [
    "DataQualityIssue",
    "DataQualityReport",
    "DatasetShard",
    "DuplicateGroup",
    "DuplicateReport",
    "ShardReport",
    "ShardResult",
    "SplitReport",
    "SplitResult",
    "analyze_dataset",
    "check_leakage",
    "find_duplicates",
    "find_near_duplicates",
    "shard_dataset",
    "split_dataset",
]
