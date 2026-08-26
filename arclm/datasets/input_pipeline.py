"""Model input preparation shared by training and fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from ..tokenizers import ChatTemplate, Tokenizer
from .transforms import record_text


@dataclass(frozen=True)
class ModelInput:
    """Tokenizer output prepared for causal language-model training."""

    text: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    chunks: list[dict[str, list[int]]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelInputPreparer:
    """Format records, tokenize them, and prepare model-input structures."""

    def __init__(self, *, chat_template: ChatTemplate | None = None):
        self.chat_template = chat_template or ChatTemplate.default()

    def format_records(self, records: Iterable[dict[str, Any]]) -> list[str]:
        return [self.format_record(record) for record in records]

    def format_record(self, record: dict[str, Any]) -> str:
        if self._is_message_list(record.get("messages")):
            return self.chat_template.format(record["messages"])
        if self._is_message_list(record.get("value")):
            return self.chat_template.format(record["value"])
        if "role" in record and "content" in record:
            return self.chat_template.format([record])
        if "prompt" in record and "completion" in record:
            return self.chat_template.format(
                [
                    {"role": "user", "content": record["prompt"]},
                    {"role": "assistant", "content": record["completion"]},
                ]
            )
        if "instruction" in record and "output" in record:
            return self.chat_template.format(
                [
                    {"role": "user", "content": record["instruction"]},
                    {"role": "assistant", "content": record["output"]},
                ]
            )
        return record_text(record)

    def prepare(
        self,
        records: Iterable[dict[str, Any]],
        *,
        tokenizer: Tokenizer,
        block_size: int,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool = False,
        pad_to_length: int | None = None,
        chunking: bool = True,
        packing: bool = False,
        loss_masking: bool = False,
    ) -> ModelInput:
        texts = self.format_records(records)
        text = "\n".join(item for item in texts if item)
        if not tokenizer.is_built:
            tokenizer.build(text)
        input_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            input_ids = input_ids[: max(0, int(max_length))]

        attention_mask = [1] * len(input_ids)
        labels = list(input_ids)
        if padding and (pad_to_length is not None or max_length is not None):
            target = int(pad_to_length or max_length or block_size)
            input_ids, attention_mask, labels = self._pad(input_ids, attention_mask, labels, tokenizer=tokenizer, target=target)

        chunks = self._chunks(
            input_ids,
            attention_mask,
            labels,
            tokenizer=tokenizer,
            block_size=block_size,
            padding=padding,
        ) if chunking else []
        return ModelInput(
            text=text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            chunks=chunks,
            metadata={
                "records": len(texts),
                "block_size": block_size,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "padding": padding,
                "packing": packing,
                "loss_masking": loss_masking,
            },
        )

    def _chunks(
        self,
        input_ids: list[int],
        attention_mask: list[int],
        labels: list[int],
        *,
        tokenizer: Tokenizer,
        block_size: int,
        padding: bool,
    ) -> list[dict[str, list[int]]]:
        chunks = []
        stride = int(block_size)
        for start in range(0, max(len(input_ids) - 1, 0), stride):
            end = start + stride
            chunk_ids = input_ids[start:end]
            chunk_labels = labels[start + 1 : end + 1]
            chunk_mask = attention_mask[start:end]
            if len(chunk_ids) < stride or len(chunk_labels) < stride:
                if not padding:
                    continue
                chunk_ids, chunk_mask, chunk_labels = self._pad(chunk_ids, chunk_mask, chunk_labels, tokenizer=tokenizer, target=stride)
            chunks.append({"input_ids": chunk_ids, "attention_mask": chunk_mask, "labels": chunk_labels})
        return chunks

    @staticmethod
    def _pad(
        input_ids: list[int],
        attention_mask: list[int],
        labels: list[int],
        *,
        tokenizer: Tokenizer,
        target: int,
    ) -> tuple[list[int], list[int], list[int]]:
        pad_id = tokenizer.special_token_id("pad") if tokenizer.is_built else 0
        label_pad = -100
        missing = max(0, target - len(input_ids))
        return (
            [*input_ids, *([pad_id] * missing)][:target],
            [*attention_mask, *([0] * missing)][:target],
            [*labels, *([label_pad] * missing)][:target],
        )

    @staticmethod
    def _is_message_list(value: Any) -> bool:
        return isinstance(value, list) and all(isinstance(item, dict) and "content" in item for item in value)


__all__ = ["ModelInput", "ModelInputPreparer"]
