"""Composable data preparation pipeline."""

from __future__ import annotations

import copy
import random
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .exceptions import DatasetError
from .schemas import DatasetValidationReport, validate_records


Record = Dict[str, Any]
RecordCallable = Callable[[Record], Optional[Record]]
FilterCallable = Callable[[Record], bool]


@dataclass(frozen=True)
class PipelineOperation:
    """Serializable description of one pipeline operation."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    serializable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return operation metadata."""

        return asdict(self)


@dataclass
class PipelineOperationResult:
    """Execution statistics for one pipeline operation."""

    name: str
    input_count: int
    output_count: int
    removed_count: int = 0
    affected_count: int = 0
    duration_seconds: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass
class DataPipelineReport:
    """Structured report for a :class:`DataPipeline` run."""

    input_count: int
    output_count: int
    removed_record_count: int
    duration_seconds: float
    operations: List[PipelineOperationResult]
    validation_failures: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    configuration: List[Dict[str, Any]] = field(default_factory=list)
    reproducibility: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return whether no validation failures or operation errors occurred."""

        return not self.validation_failures and not any(op.errors for op in self.operations)

    def summary(self) -> str:
        """Return a compact human-readable report summary."""

        return (
            f"input={self.input_count} output={self.output_count} "
            f"removed={self.removed_record_count} operations={len(self.operations)} "
            f"validation_failures={len(self.validation_failures)}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "removed_record_count": self.removed_record_count,
            "duration_seconds": self.duration_seconds,
            "operations": [operation.to_dict() for operation in self.operations],
            "validation_failures": list(self.validation_failures),
            "warnings": list(self.warnings),
            "configuration": list(self.configuration),
            "reproducibility": dict(self.reproducibility),
            "is_valid": self.is_valid,
        }


@dataclass
class _ExecutableOperation:
    metadata: PipelineOperation
    function: Callable[[List[Record]], Tuple[List[Record], int, int, List[str], List[Dict[str, Any]]]]


class DataPipeline:
    """Deterministic, composable pipeline for in-memory dataset records.

    Operations copy records by default and never mutate the user-provided input
    list unless ``copy_records=False`` is passed to :meth:`run`.
    """

    def __init__(self, *, seed: int = 42):
        self.seed = seed
        self._operations: List[_ExecutableOperation] = []

    def inspect(self) -> List[Dict[str, Any]]:
        """Return configured operation metadata before execution."""

        return [operation.metadata.to_dict() for operation in self._operations]

    def to_config(self) -> List[Dict[str, Any]]:
        """Return serializable operation configuration where possible."""

        return self.inspect()

    def strip_whitespace(self, field: str = "text") -> "DataPipeline":
        """Strip leading/trailing whitespace from a string field."""

        def run(records: List[Record]):
            affected = 0
            output = []
            for record in records:
                item = dict(record)
                value = item.get(field)
                if isinstance(value, str):
                    stripped = value.strip()
                    affected += int(stripped != value)
                    item[field] = stripped
                output.append(item)
            return output, 0, affected, [], []

        return self._add("strip_whitespace", {"field": field}, run)

    def normalize_text(self, field: str = "text") -> "DataPipeline":
        """Normalize repeated whitespace in a string field."""

        def run(records: List[Record]):
            affected = 0
            output = []
            for record in records:
                item = dict(record)
                value = item.get(field)
                if isinstance(value, str):
                    normalized = re.sub(r"\s+", " ", value).strip()
                    affected += int(normalized != value)
                    item[field] = normalized
                output.append(item)
            return output, 0, affected, [], []

        return self._add("normalize_text", {"field": field}, run)

    def remove_empty(self, field: str = "text") -> "DataPipeline":
        """Remove records whose selected field is empty."""

        def run(records: List[Record]):
            output = [
                record
                for record in records
                if str(record.get(field, "") or "").strip() != ""
            ]
            return output, len(records) - len(output), len(records) - len(output), [], []

        return self._add("remove_empty", {"field": field}, run)

    def rename_fields(self, mapping: Mapping[str, str]) -> "DataPipeline":
        """Rename fields using ``old_name -> new_name`` mapping."""

        mapping = dict(mapping)

        def run(records: List[Record]):
            affected = 0
            output = []
            for record in records:
                item = dict(record)
                for source, target in mapping.items():
                    if source in item:
                        item[target] = item.pop(source)
                        affected += 1
                output.append(item)
            return output, 0, affected, [], []

        return self._add("rename_fields", {"mapping": mapping}, run)

    def select_fields(self, fields: Sequence[str]) -> "DataPipeline":
        """Keep only selected fields."""

        selected = list(fields)

        def run(records: List[Record]):
            output = [{field_name: record[field_name] for field_name in selected if field_name in record} for record in records]
            affected = sum(1 for before, after in zip(records, output) if set(before) != set(after))
            return output, 0, affected, [], []

        return self._add("select_fields", {"fields": selected}, run)

    def join_fields(self, fields: Sequence[str], output_field: str = "text", separator: str = "\n") -> "DataPipeline":
        """Join selected fields into one output field."""

        selected = list(fields)

        def run(records: List[Record]):
            output = []
            affected = 0
            for record in records:
                item = dict(record)
                joined = separator.join(str(record.get(field_name, "")) for field_name in selected if record.get(field_name, "") != "")
                affected += int(item.get(output_field) != joined)
                item[output_field] = joined
                output.append(item)
            return output, 0, affected, [], []

        return self._add("join_fields", {"fields": selected, "output_field": output_field, "separator": separator}, run)

    def map_records(self, function: RecordCallable, *, name: str = "map_records") -> "DataPipeline":
        """Apply a custom record transformation.

        Callable operations are deterministic only if the callable is
        deterministic. They are marked as not fully serializable.
        """

        def run(records: List[Record]):
            output = []
            removed = 0
            errors: List[Dict[str, Any]] = []
            for index, record in enumerate(records):
                try:
                    value = function(dict(record))
                except Exception as exc:
                    errors.append({"index": index, "operation": name, "message": str(exc), "type": type(exc).__name__})
                    removed += 1
                    continue
                if value is None:
                    removed += 1
                else:
                    output.append(dict(value))
            return output, removed, len(records), [], errors

        return self._add(name, {"callable": repr(function)}, run, serializable=False)

    def filter_records(self, predicate: FilterCallable, *, name: str = "filter_records") -> "DataPipeline":
        """Keep records for which a predicate returns true."""

        def run(records: List[Record]):
            output = []
            errors: List[Dict[str, Any]] = []
            for index, record in enumerate(records):
                try:
                    if predicate(dict(record)):
                        output.append(record)
                except Exception as exc:
                    errors.append({"index": index, "operation": name, "message": str(exc), "type": type(exc).__name__})
            return output, len(records) - len(output), len(records) - len(output), [], errors

        return self._add(name, {"callable": repr(predicate)}, run, serializable=False)

    def deduplicate(self, field: str = "text") -> "DataPipeline":
        """Remove duplicate records using the selected field."""

        def run(records: List[Record]):
            seen = set()
            output = []
            removed = 0
            for record in records:
                key = str(record.get(field, ""))
                if key in seen:
                    removed += 1
                    continue
                seen.add(key)
                output.append(record)
            return output, removed, removed, [], []

        return self._add("deduplicate", {"field": field}, run)

    def validate(self, schema: str, *, strict: bool = True, allow_empty: bool = False) -> "DataPipeline":
        """Validate records and keep records unchanged."""

        def run(records: List[Record]):
            report = validate_records(records, schema=schema, strict=strict, allow_empty=allow_empty)
            warnings = [issue.message for issue in report.warnings]
            return records, 0, 0, warnings, [issue.to_dict() for issue in report.errors]

        return self._add("validate", {"schema": schema, "strict": strict, "allow_empty": allow_empty}, run)

    def format_prompt_completion(self, output_field: str = "text") -> "DataPipeline":
        """Format prompt/completion rows into a text field."""

        def formatter(record: Record) -> Record:
            item = dict(record)
            item[output_field] = f"Prompt: {record.get('prompt', '')}\nCompletion: {record.get('completion', '')}".strip()
            return item

        return self.map_records(formatter, name="format_prompt_completion")

    def format_instruction(self, output_field: str = "text") -> "DataPipeline":
        """Format instruction/input/output rows into a text field."""

        def formatter(record: Record) -> Record:
            parts = [str(record.get("instruction", "")).strip()]
            if str(record.get("input", "")).strip():
                parts.append(str(record.get("input", "")).strip())
            parts.append(str(record.get("output", "")).strip())
            item = dict(record)
            item[output_field] = "\n".join(part for part in parts if part)
            return item

        return self.map_records(formatter, name="format_instruction")

    def format_conversations(self, output_field: str = "text") -> "DataPipeline":
        """Format conversation messages into a text field."""

        def formatter(record: Record) -> Record:
            lines = []
            for message in record.get("messages", []):
                role = str(message.get("role", "")).strip()
                content = str(message.get("content", "")).strip()
                if role or content:
                    lines.append(f"{role}: {content}".strip())
            item = dict(record)
            item[output_field] = "\n".join(lines)
            return item

        return self.map_records(formatter, name="format_conversations")

    def apply_tokenizer(self, tokenizer: Any, field: str = "text", output_field: str = "tokens") -> "DataPipeline":
        """Add tokenizer output to each record."""

        def run(records: List[Record]):
            output = []
            errors: List[Dict[str, Any]] = []
            for index, record in enumerate(records):
                item = dict(record)
                try:
                    item[output_field] = tokenizer.encode_text(str(item.get(field, "")))
                except Exception as exc:
                    errors.append({"index": index, "operation": "apply_tokenizer", "message": str(exc), "type": type(exc).__name__})
                output.append(item)
            return output, 0, len(records), [], errors

        return self._add("apply_tokenizer", {"field": field, "output_field": output_field, "tokenizer": type(tokenizer).__name__}, run, serializable=False)

    def split(self, train: float = 0.8, validation: float = 0.1, test: float = 0.1) -> "DataPipeline":
        """Shuffle deterministically and return one record with split lists."""

        def run(records: List[Record]):
            total = train + validation + test
            if total <= 0:
                raise DatasetError("At least one split ratio must be greater than zero.")
            shuffled = list(records)
            random.Random(self.seed).shuffle(shuffled)
            train_count = int(len(shuffled) * (train / total))
            validation_count = int(len(shuffled) * (validation / total))
            split_record = {
                "train": shuffled[:train_count],
                "validation": shuffled[train_count:train_count + validation_count],
                "test": shuffled[train_count + validation_count:],
            }
            return [split_record], 0, len(records), [], []

        return self._add("split", {"train": train, "validation": validation, "test": test, "seed": self.seed}, run)

    def run(self, records: Iterable[Mapping[str, Any]], *, copy_records: bool = True) -> Tuple[List[Record], DataPipelineReport]:
        """Execute the configured pipeline."""

        started = time.perf_counter()
        current = [copy.deepcopy(dict(record)) if copy_records else dict(record) for record in records]
        input_count = len(current)
        operation_results: List[PipelineOperationResult] = []
        validation_failures: List[Dict[str, Any]] = []
        all_warnings: List[str] = []

        for operation in self._operations:
            op_started = time.perf_counter()
            before_count = len(current)
            try:
                current, removed, affected, warnings, errors = operation.function(current)
            except Exception as exc:
                raise DatasetError(f"DataPipeline operation {operation.metadata.name!r} failed: {exc}") from exc
            elapsed = time.perf_counter() - op_started
            if operation.metadata.name == "validate":
                validation_failures.extend(errors)
            all_warnings.extend(warnings)
            operation_results.append(
                PipelineOperationResult(
                    name=operation.metadata.name,
                    input_count=before_count,
                    output_count=len(current),
                    removed_count=removed,
                    affected_count=affected,
                    duration_seconds=elapsed,
                    errors=errors,
                    warnings=warnings,
                )
            )

        report = DataPipelineReport(
            input_count=input_count,
            output_count=len(current),
            removed_record_count=input_count - len(current),
            duration_seconds=time.perf_counter() - started,
            operations=operation_results,
            validation_failures=validation_failures,
            warnings=all_warnings,
            configuration=self.to_config(),
            reproducibility={"seed": self.seed, "deterministic": True},
        )
        return current, report

    def _add(
        self,
        name: str,
        params: Dict[str, Any],
        function: Callable[[List[Record]], Tuple[List[Record], int, int, List[str], List[Dict[str, Any]]]],
        *,
        serializable: bool = True,
    ) -> "DataPipeline":
        self._operations.append(
            _ExecutableOperation(
                metadata=PipelineOperation(name=name, params=params, serializable=serializable),
                function=function,
            )
        )
        return self


__all__ = [
    "DataPipeline",
    "DataPipelineReport",
    "PipelineOperation",
    "PipelineOperationResult",
]
