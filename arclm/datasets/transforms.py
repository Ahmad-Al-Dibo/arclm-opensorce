"""Composable dataset transformations."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol


class Transform(Protocol):
    """Research extension contract for dataset transforms."""

    def __call__(self, record: dict[str, Any]) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class FieldMapTransform:
    """Rename or project fields."""

    mapping: dict[str, str]

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        updated = dict(record)
        for source, target in self.mapping.items():
            if source in updated:
                updated[target] = updated.pop(source)
        return updated


@dataclass(frozen=True)
class FunctionTransform:
    """Wrap a callable as a Transform."""

    function: Callable[[dict[str, Any]], dict[str, Any] | None]

    def __call__(self, record: dict[str, Any]) -> dict[str, Any] | None:
        return self.function(dict(record))


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def record_text(record: dict[str, Any]) -> str:
    if "text" in record:
        return str(record["text"])
    if "prompt" in record and "completion" in record:
        return f"{record['prompt']} {record['completion']}"
    if "instruction" in record and "output" in record:
        return f"{record['instruction']} {record['output']}"
    return " ".join(str(value) for key, value in record.items() if not str(key).startswith("_"))


def clean_records(
    records: list[dict[str, Any]],
    *,
    remove_empty: bool = True,
    remove_invalid: bool = True,
    deduplicate: bool = False,
    normalize_space: bool = False,
    lowercase: bool = False,
    text_fields: tuple[str, ...] | None = None,
    character_filter: Callable[[str], bool] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    removed_empty = 0
    removed_invalid = 0
    removed_duplicates = 0
    fields = text_fields

    for record in records:
        if remove_invalid and not isinstance(record, dict):
            removed_invalid += 1
            continue
        updated = dict(record)
        target_fields = fields or tuple(key for key, value in updated.items() if isinstance(value, str) and not str(key).startswith("_"))
        for field in target_fields:
            if field in updated and isinstance(updated[field], str):
                value = updated[field]
                if normalize_space:
                    value = normalize_whitespace(value)
                if lowercase:
                    value = value.lower()
                if character_filter is not None:
                    value = "".join(character for character in value if character_filter(character))
                updated[field] = value
        text = record_text(updated)
        if remove_empty and not text.strip():
            removed_empty += 1
            continue
        fingerprint = text if text else repr(sorted(updated.items()))
        if deduplicate and fingerprint in seen:
            removed_duplicates += 1
            continue
        seen.add(fingerprint)
        cleaned.append(updated)

    return cleaned, {
        "removed_empty": removed_empty,
        "removed_invalid": removed_invalid,
        "removed_duplicates": removed_duplicates,
        "output_records": len(cleaned),
    }


def shuffled(records: list[dict[str, Any]], *, seed: int | None = None) -> list[dict[str, Any]]:
    values = [dict(record) for record in records]
    rng = random.Random(seed)
    rng.shuffle(values)
    return values


__all__ = ["FieldMapTransform", "FunctionTransform", "Transform", "clean_records", "normalize_whitespace", "record_text", "shuffled"]
