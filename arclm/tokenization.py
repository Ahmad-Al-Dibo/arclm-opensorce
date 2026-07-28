"""Production-oriented tokenization workflows and cache integration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

from ._version import __version__
from .cache import read_cache, write_cache
from .exceptions import OptionalDependencyError
from .reproducibility import fingerprint


Record = dict[str, Any]


@dataclass(frozen=True)
class TokenizationConfig:
    """Serializable tokenization configuration."""

    tokenizer: str
    schema: str = "text"
    text_field: str = "text"
    max_length: Optional[int] = None
    truncation: bool = True
    padding: bool | str = False
    add_eos: bool = False
    add_bos: bool = False
    create_labels: bool = True
    prompt_masking: bool = False
    cache_dir: Optional[str] = None
    tokenizer_revision: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizedDataset:
    """Tokenized dataset plus cache metadata."""

    records: list[dict[str, Any]]
    config: TokenizationConfig
    cache_key: str
    cache_hit: bool
    tokenizer_identity: str
    dataset_fingerprint: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    report_type: str = "tokenized_dataset"
    schema_version: str = "1.0"
    arclm_version: str = __version__

    def __iter__(self):
        yield from self.records

    def __len__(self) -> int:
        return len(self.records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "config": self.config.to_dict(),
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "tokenizer_identity": self.tokenizer_identity,
            "dataset_fingerprint": self.dataset_fingerprint,
            "created_at": self.created_at,
            "report_type": self.report_type,
            "schema_version": self.schema_version,
            "arclm_version": self.arclm_version,
        }


def tokenize_dataset(
    dataset: Iterable[Mapping[str, Any]],
    *,
    tokenizer: str | Any,
    schema: str = "text",
    text_field: str = "text",
    max_length: Optional[int] = None,
    truncation: bool = True,
    padding: bool | str = False,
    batch_size: int = 32,
    cache_dir: Optional[str] = None,
    read_only_cache: bool = False,
    add_eos: bool = False,
    add_bos: bool = False,
    create_labels: bool = True,
    prompt_masking: bool = False,
    tokenizer_revision: Optional[str] = None,
) -> TokenizedDataset:
    """Format and tokenize records for causal language-model workflows."""

    rows = [dict(row) for row in dataset]
    tokenizer_obj, tokenizer_identity = _load_tokenizer(tokenizer)
    config = TokenizationConfig(
        tokenizer=tokenizer_identity,
        schema=schema,
        text_field=text_field,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        add_eos=add_eos,
        add_bos=add_bos,
        create_labels=create_labels,
        prompt_masking=prompt_masking,
        cache_dir=cache_dir,
        tokenizer_revision=tokenizer_revision,
    )
    dataset_fp = fingerprint(rows).value
    cache_key = _cache_key(dataset_fp, config)
    if cache_dir:
        cached = read_cache(cache_dir, cache_key, read_only=read_only_cache)
        if cached is not None:
            return TokenizedDataset(
                records=list(cached["data"]["records"]),
                config=config,
                cache_key=cache_key,
                cache_hit=True,
                tokenizer_identity=tokenizer_identity,
                dataset_fingerprint=dataset_fp,
            )

    tokenized: list[dict[str, Any]] = []
    for start in range(0, len(rows), max(1, batch_size)):
        batch = rows[start : start + max(1, batch_size)]
        tokenized.extend(
            _tokenize_one(
                row,
                tokenizer_obj,
                schema=schema,
                text_field=text_field,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                add_eos=add_eos,
                add_bos=add_bos,
                create_labels=create_labels,
                prompt_masking=prompt_masking,
            )
            for row in batch
        )

    result = TokenizedDataset(tokenized, config, cache_key, False, tokenizer_identity, dataset_fp)
    if cache_dir:
        write_cache(cache_dir, cache_key, {"records": tokenized}, config=config.to_dict(), read_only=read_only_cache)
    return result


def _load_tokenizer(tokenizer: str | Any) -> tuple[Any, str]:
    if isinstance(tokenizer, str):
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            raise OptionalDependencyError("Transformers is required to load tokenizer by name.") from exc
        return AutoTokenizer.from_pretrained(tokenizer), tokenizer
    identity = getattr(tokenizer, "name_or_path", None) or type(tokenizer).__name__
    return tokenizer, str(identity)


def _tokenize_one(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    schema: str,
    text_field: str,
    max_length: Optional[int],
    truncation: bool,
    padding: bool | str,
    add_eos: bool,
    add_bos: bool,
    create_labels: bool,
    prompt_masking: bool,
) -> dict[str, Any]:
    prompt, text = _format_record(row, schema=schema, text_field=text_field)
    if add_bos:
        bos = getattr(tokenizer, "bos_token", None)
        if bos:
            text = bos + text
    if add_eos:
        eos = getattr(tokenizer, "eos_token", None)
        if eos and not text.endswith(eos):
            text = text + eos
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
    )
    input_ids = list(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids)
    attention_mask = list(
        encoded.get("attention_mask", [1] * len(input_ids)) if isinstance(encoded, Mapping) else getattr(encoded, "attention_mask", [1] * len(input_ids))
    )
    item = {"input_ids": input_ids, "attention_mask": attention_mask}
    if create_labels:
        labels = list(input_ids)
        if prompt_masking and prompt:
            prompt_ids = tokenizer(prompt, max_length=max_length, truncation=truncation, padding=False)["input_ids"]
            for index in range(min(len(prompt_ids), len(labels))):
                labels[index] = -100
        item["labels"] = labels
    item["length"] = len(input_ids)
    return item


def _format_record(row: Mapping[str, Any], *, schema: str, text_field: str) -> tuple[str, str]:
    if schema == "text":
        text = str(row.get(text_field, ""))
        return "", text
    if schema == "prompt_completion":
        prompt = f"Prompt: {row.get('prompt', '')}\nCompletion:"
        return prompt, f"{prompt} {row.get('completion', '')}".strip()
    if schema == "instruction":
        prompt = str(row.get("instruction", ""))
        if str(row.get("input", "")).strip():
            prompt = f"{prompt}\n{row.get('input', '')}"
        return prompt, f"{prompt}\n{row.get('output', '')}".strip()
    if schema == "conversation":
        lines: list[str] = []
        prompt_lines: list[str] = []
        for message in row.get("messages", []):
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            line = f"{role}: {content}".strip()
            lines.append(line)
            if role != "assistant":
                prompt_lines.append(line)
        return "\n".join(prompt_lines), "\n".join(lines)
    raise ValueError(f"Unsupported tokenization schema: {schema}")


def _cache_key(dataset_fingerprint: str, config: TokenizationConfig) -> str:
    payload = json.dumps(
        {
            "dataset": dataset_fingerprint,
            "config": config.to_dict(),
            "arclm_version": __version__,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["TokenizationConfig", "TokenizedDataset", "tokenize_dataset"]
