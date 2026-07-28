"""Formal dataset schemas and validation reports."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

from .exceptions import DatasetValidationError


VALID_SCHEMAS = {"text", "prompt_completion", "instruction", "conversation"}
CONVERSATION_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class ValidationIssue:
    """A validation error or warning for one record."""

    index: int
    schema: str
    field: str
    category: str
    message: str
    level: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        """Return the issue as a dictionary."""

        return asdict(self)


@dataclass
class BaseRecord:
    """Base record with optional metadata."""

    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_name: ClassVar[str] = "base"
    allowed_fields: ClassVar[Tuple[str, ...]] = ("metadata",)
    required_fields: ClassVar[Tuple[str, ...]] = ()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary, omitting empty metadata."""

        data = asdict(self)
        if not data.get("metadata"):
            data.pop("metadata", None)
        return data

    @classmethod
    def validate(
        cls,
        record: Mapping[str, Any],
        *,
        index: int = 0,
        strict: bool = True,
        allow_empty: bool = False,
    ) -> Tuple[Optional["BaseRecord"], List[ValidationIssue], List[ValidationIssue]]:
        """Validate a mapping and return record, errors, and warnings."""

        raise NotImplementedError

    @classmethod
    def _common_checks(
        cls,
        record: Mapping[str, Any],
        *,
        index: int,
        strict: bool,
    ) -> Tuple[List[ValidationIssue], List[ValidationIssue]]:
        errors: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []
        if not isinstance(record, Mapping):
            return [
                ValidationIssue(
                    index=index,
                    schema=cls.schema_name,
                    field="<record>",
                    category="type_error",
                    message="Record must be a mapping/dictionary.",
                )
            ], warnings

        for field_name in cls.required_fields:
            if field_name not in record:
                errors.append(
                    ValidationIssue(
                        index=index,
                        schema=cls.schema_name,
                        field=field_name,
                        category="missing_required_field",
                        message=f"Missing required field: {field_name}.",
                    )
                )

        unknown = sorted(set(record) - set(cls.allowed_fields))
        if unknown:
            issue = ValidationIssue(
                index=index,
                schema=cls.schema_name,
                field=",".join(unknown),
                category="unknown_field",
                message="Unknown field(s): " + ", ".join(unknown) + ".",
                level="error" if strict else "warning",
            )
            (errors if strict else warnings).append(issue)

        metadata = record.get("metadata", {})
        if "metadata" in record and not isinstance(metadata, Mapping):
            errors.append(
                ValidationIssue(
                    index=index,
                    schema=cls.schema_name,
                    field="metadata",
                    category="type_error",
                    message="metadata must be a dictionary when provided.",
                )
            )
        return errors, warnings

    @classmethod
    def _string_field(
        cls,
        record: Mapping[str, Any],
        field_name: str,
        *,
        index: int,
        errors: List[ValidationIssue],
        warnings: List[ValidationIssue],
        allow_empty: bool,
        required: bool = True,
    ) -> str:
        value = record.get(field_name, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            errors.append(
                ValidationIssue(
                    index=index,
                    schema=cls.schema_name,
                    field=field_name,
                    category="type_error",
                    message=f"{field_name} must be a string.",
                )
            )
            return ""
        if required and not allow_empty and value.strip() == "":
            errors.append(
                ValidationIssue(
                    index=index,
                    schema=cls.schema_name,
                    field=field_name,
                    category="empty_required_field",
                    message=f"{field_name} must not be empty.",
                )
            )
        elif value.strip() == "":
            warnings.append(
                ValidationIssue(
                    index=index,
                    schema=cls.schema_name,
                    field=field_name,
                    category="empty_optional_field",
                    message=f"{field_name} is empty.",
                    level="warning",
                )
            )
        return value


@dataclass
class TextRecord(BaseRecord):
    """A pretraining text record."""

    text: str = ""

    schema_name: ClassVar[str] = "text"
    allowed_fields: ClassVar[Tuple[str, ...]] = ("text", "metadata")
    required_fields: ClassVar[Tuple[str, ...]] = ("text",)

    @classmethod
    def validate(cls, record: Mapping[str, Any], *, index: int = 0, strict: bool = True, allow_empty: bool = False):
        errors, warnings = cls._common_checks(record, index=index, strict=strict)
        if errors and not isinstance(record, Mapping):
            return None, errors, warnings
        text = cls._string_field(record, "text", index=index, errors=errors, warnings=warnings, allow_empty=allow_empty)
        if errors:
            return None, errors, warnings
        return cls(text=text, metadata=dict(record.get("metadata", {}) or {})), errors, warnings


@dataclass
class PromptCompletionRecord(BaseRecord):
    """A prompt-completion fine-tuning record."""

    prompt: str = ""
    completion: str = ""

    schema_name: ClassVar[str] = "prompt_completion"
    allowed_fields: ClassVar[Tuple[str, ...]] = ("prompt", "completion", "metadata")
    required_fields: ClassVar[Tuple[str, ...]] = ("prompt", "completion")

    @classmethod
    def validate(cls, record: Mapping[str, Any], *, index: int = 0, strict: bool = True, allow_empty: bool = False):
        errors, warnings = cls._common_checks(record, index=index, strict=strict)
        if errors and not isinstance(record, Mapping):
            return None, errors, warnings
        prompt = cls._string_field(record, "prompt", index=index, errors=errors, warnings=warnings, allow_empty=allow_empty)
        completion = cls._string_field(record, "completion", index=index, errors=errors, warnings=warnings, allow_empty=allow_empty)
        if errors:
            return None, errors, warnings
        return cls(prompt=prompt, completion=completion, metadata=dict(record.get("metadata", {}) or {})), errors, warnings


@dataclass
class InstructionRecord(BaseRecord):
    """An instruction/input/output supervised fine-tuning record."""

    instruction: str = ""
    input: str = ""
    output: str = ""

    schema_name: ClassVar[str] = "instruction"
    allowed_fields: ClassVar[Tuple[str, ...]] = ("instruction", "input", "output", "metadata")
    required_fields: ClassVar[Tuple[str, ...]] = ("instruction", "output")

    @classmethod
    def validate(cls, record: Mapping[str, Any], *, index: int = 0, strict: bool = True, allow_empty: bool = False):
        errors, warnings = cls._common_checks(record, index=index, strict=strict)
        if errors and not isinstance(record, Mapping):
            return None, errors, warnings
        instruction = cls._string_field(record, "instruction", index=index, errors=errors, warnings=warnings, allow_empty=allow_empty)
        input_text = cls._string_field(record, "input", index=index, errors=errors, warnings=warnings, allow_empty=True, required=False)
        output = cls._string_field(record, "output", index=index, errors=errors, warnings=warnings, allow_empty=allow_empty)
        if errors:
            return None, errors, warnings
        return cls(instruction=instruction, input=input_text, output=output, metadata=dict(record.get("metadata", {}) or {})), errors, warnings


@dataclass
class MessageRecord:
    """One conversation message."""

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Return a plain dictionary."""

        return {"role": self.role, "content": self.content}


@dataclass
class ConversationRecord(BaseRecord):
    """A chat conversation record with role/content messages."""

    messages: List[MessageRecord] = field(default_factory=list)

    schema_name: ClassVar[str] = "conversation"
    allowed_fields: ClassVar[Tuple[str, ...]] = ("messages", "metadata")
    required_fields: ClassVar[Tuple[str, ...]] = ("messages",)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""

        data: Dict[str, Any] = {"messages": [message.to_dict() for message in self.messages]}
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def validate(cls, record: Mapping[str, Any], *, index: int = 0, strict: bool = True, allow_empty: bool = False):
        errors, warnings = cls._common_checks(record, index=index, strict=strict)
        if errors and not isinstance(record, Mapping):
            return None, errors, warnings
        raw_messages = record.get("messages", [])
        if not isinstance(raw_messages, list):
            errors.append(
                ValidationIssue(index, cls.schema_name, "messages", "type_error", "messages must be a list.")
            )
            return None, errors, warnings
        if not raw_messages and not allow_empty:
            errors.append(
                ValidationIssue(index, cls.schema_name, "messages", "empty_required_field", "messages must not be empty.")
            )
        messages: List[MessageRecord] = []
        for message_index, message in enumerate(raw_messages):
            prefix = f"messages[{message_index}]"
            if not isinstance(message, Mapping):
                errors.append(
                    ValidationIssue(index, cls.schema_name, prefix, "type_error", f"{prefix} must be a dictionary.")
                )
                continue
            role = str(message.get("role", "")).lower().strip()
            content = message.get("content", "")
            if role not in CONVERSATION_ROLES:
                errors.append(
                    ValidationIssue(
                        index,
                        cls.schema_name,
                        f"{prefix}.role",
                        "invalid_role",
                        f"{prefix}.role must be one of: {', '.join(sorted(CONVERSATION_ROLES))}.",
                    )
                )
            if not isinstance(content, str):
                errors.append(
                    ValidationIssue(index, cls.schema_name, f"{prefix}.content", "type_error", f"{prefix}.content must be a string.")
                )
                content = ""
            if not allow_empty and str(content).strip() == "":
                errors.append(
                    ValidationIssue(index, cls.schema_name, f"{prefix}.content", "empty_required_field", f"{prefix}.content must not be empty.")
                )
            message_unknown = sorted(set(message) - {"role", "content"})
            if message_unknown:
                issue = ValidationIssue(
                    index,
                    cls.schema_name,
                    prefix,
                    "unknown_field",
                    f"{prefix} has unknown field(s): {', '.join(message_unknown)}.",
                    level="error" if strict else "warning",
                )
                (errors if strict else warnings).append(issue)
            if role in CONVERSATION_ROLES:
                messages.append(MessageRecord(role=role, content=str(content)))
        if errors:
            return None, errors, warnings
        return cls(messages=messages, metadata=dict(record.get("metadata", {}) or {})), errors, warnings


SCHEMA_CLASSES: Dict[str, Type[BaseRecord]] = {
    "text": TextRecord,
    "prompt_completion": PromptCompletionRecord,
    "prompt-completion": PromptCompletionRecord,
    "instruction": InstructionRecord,
    "conversation": ConversationRecord,
}


@dataclass
class DatasetValidationReport:
    """Structured result for dataset validation."""

    schema: str
    strict: bool
    total_records: int
    valid_records: int
    invalid_records: int
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    invalid_record_indexes: List[int] = field(default_factory=list)
    detected_fields: List[str] = field(default_factory=list)
    empty_content: Dict[str, int] = field(default_factory=dict)
    duplicates: Dict[str, Any] = field(default_factory=dict)
    length_stats: Dict[str, Any] = field(default_factory=dict)
    validated_records: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return whether all records were valid."""

        return self.invalid_records == 0 and not self.errors

    @property
    def error_count(self) -> int:
        """Return the number of validation errors."""

        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """Return the number of validation warnings."""

        return len(self.warnings)

    @property
    def error_categories(self) -> Dict[str, int]:
        """Return counts by error category."""

        return dict(Counter(issue.category for issue in self.errors))

    def summary(self) -> str:
        """Return a compact human-readable summary."""

        return (
            f"schema={self.schema} total={self.total_records} valid={self.valid_records} "
            f"invalid={self.invalid_records} errors={self.error_count} warnings={self.warning_count}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable report."""

        return {
            "schema": self.schema,
            "strict": self.strict,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "warning_count": self.warning_count,
            "error_count": self.error_count,
            "error_categories": self.error_categories,
            "invalid_record_indexes": list(self.invalid_record_indexes),
            "sample_errors": [issue.to_dict() for issue in self.errors[:10]],
            "sample_warnings": [issue.to_dict() for issue in self.warnings[:10]],
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "detected_fields": list(self.detected_fields),
            "empty_content": dict(self.empty_content),
            "duplicates": dict(self.duplicates),
            "length_stats": dict(self.length_stats),
            "validated_records": list(self.validated_records),
        }

    def raise_for_errors(self) -> None:
        """Raise :class:`DatasetValidationError` when errors exist."""

        if self.errors:
            first = self.errors[0]
            raise DatasetValidationError(
                f"{self.error_count} validation error(s); first error at record "
                f"{first.index}, field {first.field}: {first.message}"
            )


def normalize_schema_name(schema: str) -> str:
    """Normalize a schema name and validate it."""

    normalized = str(schema).lower().replace("-", "_").strip()
    if normalized not in VALID_SCHEMAS:
        raise DatasetValidationError(
            "schema must be one of: " + ", ".join(sorted(VALID_SCHEMAS))
        )
    return normalized


def validate_record(
    record: Mapping[str, Any],
    *,
    schema: str,
    index: int = 0,
    strict: bool = True,
    allow_empty: bool = False,
) -> Tuple[Optional[BaseRecord], List[ValidationIssue], List[ValidationIssue]]:
    """Validate one record against a named schema."""

    normalized = normalize_schema_name(schema)
    cls = SCHEMA_CLASSES[normalized]
    return cls.validate(record, index=index, strict=strict, allow_empty=allow_empty)


def validate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    schema: str,
    strict: bool = True,
    allow_empty: bool = False,
    check_duplicates: bool = False,
    duplicate_field: Optional[str] = None,
) -> DatasetValidationReport:
    """Validate a batch of records and return a structured report."""

    normalized = normalize_schema_name(schema)
    rows = list(records)
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []
    invalid_indexes: List[int] = []
    detected_fields = sorted({str(field_name) for row in rows if isinstance(row, Mapping) for field_name in row.keys()})
    empty_content: Counter[str] = Counter()
    lengths: List[int] = []
    valid_records: List[Dict[str, Any]] = []
    seen: Counter[str] = Counter()

    for index, row in enumerate(rows):
        if isinstance(row, Mapping):
            for field_name, value in row.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    empty_content[str(field_name)] += 1
        validated, row_errors, row_warnings = validate_record(
            row,
            schema=normalized,
            index=index,
            strict=strict,
            allow_empty=allow_empty,
        )
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        if row_errors:
            invalid_indexes.append(index)
            continue
        if validated is not None:
            plain = validated.to_dict()
            valid_records.append(plain)
            text = _record_text_for_stats(plain, normalized, duplicate_field)
            if text is not None:
                lengths.append(len(text))
                if check_duplicates:
                    seen[text] += 1

    duplicate_count = sum(count - 1 for count in seen.values() if count > 1)
    duplicates = {
        "enabled": check_duplicates,
        "field": duplicate_field,
        "duplicate_records": duplicate_count,
        "unique_values": len(seen),
    }
    length_stats = _length_stats(lengths)
    return DatasetValidationReport(
        schema=normalized,
        strict=strict,
        total_records=len(rows),
        valid_records=len(valid_records),
        invalid_records=len(set(invalid_indexes)),
        errors=errors,
        warnings=warnings,
        invalid_record_indexes=sorted(set(invalid_indexes)),
        detected_fields=detected_fields,
        empty_content=dict(empty_content),
        duplicates=duplicates,
        length_stats=length_stats,
        validated_records=valid_records,
    )


def _record_text_for_stats(record: Mapping[str, Any], schema: str, duplicate_field: Optional[str]) -> Optional[str]:
    if duplicate_field:
        value = record.get(duplicate_field)
        return str(value) if value is not None else None
    if schema == "text":
        return str(record.get("text", ""))
    if schema == "prompt_completion":
        return str(record.get("prompt", "")) + "\n" + str(record.get("completion", ""))
    if schema == "instruction":
        return str(record.get("instruction", "")) + "\n" + str(record.get("input", "")) + "\n" + str(record.get("output", ""))
    if schema == "conversation":
        return "\n".join(str(message.get("content", "")) for message in record.get("messages", []))
    return None


def _length_stats(lengths: Sequence[int]) -> Dict[str, Any]:
    if not lengths:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": mean(lengths),
    }


__all__ = [
    "BaseRecord",
    "CONVERSATION_ROLES",
    "ConversationRecord",
    "DatasetValidationReport",
    "InstructionRecord",
    "MessageRecord",
    "PromptCompletionRecord",
    "TextRecord",
    "VALID_SCHEMAS",
    "ValidationIssue",
    "normalize_schema_name",
    "validate_record",
    "validate_records",
]
