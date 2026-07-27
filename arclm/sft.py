"""Public supervised fine-tuning helpers."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .data_processor import DataProcessor


@dataclass
class SFTTrainingResult:
    """Result returned by ``train_sft``."""

    model: str
    dataset: str
    output_dir: str
    backend: str
    use_lora: bool
    assistant_only_loss: bool
    train_loss_history: List[float]
    steps: int
    metadata_path: str
    adapter_path: Optional[str] = None
    full_model_path: Optional[str] = None


def train_sft(
    model: str,
    dataset: str,
    output_dir: str,
    backend: str = "huggingface",
    assistant_only_loss: bool = True,
    use_lora: bool = False,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    max_length: int = 1024,
    dtype: str = "auto",
    device_map: Optional[str] = None,
    trust_remote_code: bool = True,
    enable_thinking: Optional[bool] = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Sequence[str]] = None,
    max_steps: Optional[int] = None,
    seed: int = 42,
    save_tokenizer: bool = True,
) -> SFTTrainingResult:
    """Run supervised fine-tuning through an ArcLM backend.

    The implemented backend is ``backend="huggingface"``. It trains a
    Hugging Face causal language model with ArcLM's SFT data handling,
    optional assistant-only labels, and optional PEFT LoRA adapters.
    """

    normalized_backend = backend.lower().strip()
    if normalized_backend != "huggingface":
        raise ValueError(
            "train_sft currently implements backend='huggingface' only. "
            "ArcLM-native SFT remains available through InstructionDataset "
            "and Trainer for local ArcLM checkpoints."
        )

    return _train_sft_huggingface(
        model=model,
        dataset=dataset,
        output_dir=output_dir,
        assistant_only_loss=assistant_only_loss,
        use_lora=use_lora,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_length=max_length,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        enable_thinking=enable_thinking,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        max_steps=max_steps,
        seed=seed,
        save_tokenizer=save_tokenizer,
    )


def _train_sft_huggingface(**kwargs) -> SFTTrainingResult:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "train_sft(backend='huggingface') requires transformers. "
            "Install it with: pip install 'transformers>=4.51,<6'"
        ) from exc

    model_name = kwargs["model"]
    dataset_path = kwargs["dataset"]
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device_map = _normalize_device_map(kwargs["device_map"])

    _set_seed(kwargs["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=kwargs["trust_remote_code"],
    )
    _ensure_padding_token(tokenizer)

    model_kwargs = {"trust_remote_code": kwargs["trust_remote_code"]}
    resolved_dtype = _resolve_dtype(kwargs["dtype"])
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        if "torch_dtype" in model_kwargs:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device_map is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_model.to(device)

    if hasattr(hf_model.config, "use_cache"):
        hf_model.config.use_cache = False

    if kwargs["use_lora"]:
        hf_model = _apply_lora(
            hf_model,
            r=kwargs["lora_r"],
            alpha=kwargs["lora_alpha"],
            dropout=kwargs["lora_dropout"],
            target_modules=kwargs["lora_target_modules"],
        )

    records = _load_sft_records(dataset_path)
    train_dataset = _HuggingFaceSFTDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=kwargs["max_length"],
        assistant_only_loss=kwargs["assistant_only_loss"],
        enable_thinking=kwargs["enable_thinking"],
    )
    collator = _SFTDataCollator(tokenizer)
    loader = DataLoader(
        train_dataset,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        collate_fn=collator,
    )

    trainable_params = [param for param in hf_model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for SFT.")

    optimizer = torch.optim.AdamW(trainable_params, lr=kwargs["learning_rate"])
    loss_history = _run_training_loop(
        model=hf_model,
        loader=loader,
        optimizer=optimizer,
        num_epochs=kwargs["num_epochs"],
        gradient_accumulation_steps=kwargs["gradient_accumulation_steps"],
        max_steps=kwargs["max_steps"],
    )

    if kwargs["use_lora"]:
        hf_model.save_pretrained(output_dir)
        adapter_path = str(output_dir)
        full_model_path = None
    else:
        hf_model.save_pretrained(output_dir)
        adapter_path = None
        full_model_path = str(output_dir)

    if kwargs["save_tokenizer"]:
        tokenizer.save_pretrained(output_dir)

    metadata_path = output_dir / "arclm_sft_metadata.json"
    result = SFTTrainingResult(
        model=model_name,
        dataset=str(dataset_path),
        output_dir=str(output_dir),
        backend="huggingface",
        use_lora=kwargs["use_lora"],
        assistant_only_loss=kwargs["assistant_only_loss"],
        train_loss_history=loss_history,
        steps=len(loss_history),
        adapter_path=adapter_path,
        full_model_path=full_model_path,
        metadata_path=str(metadata_path),
    )
    metadata_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return result


class _HuggingFaceSFTDataset(Dataset):
    def __init__(
        self,
        records: List[List[Dict[str, str]]],
        tokenizer: Any,
        max_length: int,
        assistant_only_loss: bool,
        enable_thinking: Optional[bool],
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_only_loss = assistant_only_loss
        self.enable_thinking = enable_thinking

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        messages = self.records[index]
        return _encode_messages(
            tokenizer=self.tokenizer,
            messages=messages,
            max_length=self.max_length,
            assistant_only_loss=self.assistant_only_loss,
            enable_thinking=self.enable_thinking,
        )


class _SFTDataCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }


def _run_training_loop(
    model,
    loader,
    optimizer,
    num_epochs: int,
    gradient_accumulation_steps: int,
    max_steps: Optional[int],
) -> List[float]:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be greater than zero.")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than zero.")

    model.train()
    loss_history = []
    completed_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        for batch_index, batch in enumerate(loader, start=1):
            device = _model_input_device(model)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / gradient_accumulation_steps
            loss.backward()

            should_step = batch_index % gradient_accumulation_steps == 0
            should_step = should_step or batch_index == len(loader)
            if should_step:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                loss_history.append(float(raw_loss.detach().cpu().item()))
                print(
                    f"SFT epoch {epoch + 1}/{num_epochs} | "
                    f"step {completed_steps} | loss {loss_history[-1]:.4f}",
                    flush=True,
                )
                if max_steps is not None and completed_steps >= max_steps:
                    return loss_history

    return loss_history


def _load_sft_records(path: str) -> List[List[Dict[str, str]]]:
    dataset = DataProcessor.load(path).clean()
    records = []
    for row in dataset.samples:
        messages = _row_to_messages(row)
        if messages:
            records.append(messages)
    if not records:
        raise ValueError(f"No SFT records found in {path}.")
    return records


def _row_to_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(row.get("messages"), list):
        return [_normalize_message(message) for message in row["messages"]]

    if isinstance(row.get("conversations"), list):
        converted = []
        for message in row["conversations"]:
            role = str(message.get("from", "")).lower()
            if role in {"human", "user"}:
                role = "user"
            elif role in {"gpt", "assistant"}:
                role = "assistant"
            converted.append({"role": role, "content": str(message.get("value", ""))})
        return [message for message in converted if message["role"] and message["content"]]

    instruction = (
        row.get("instruction")
        or row.get("prompt")
        or row.get("question")
        or row.get("input")
        or ""
    )
    if row.get("instruction") and row.get("input"):
        instruction = f"{instruction}\n{row.get('input')}".strip()
    response = (
        row.get("output")
        or row.get("response")
        or row.get("completion")
        or row.get("answer")
        or ""
    )
    if instruction and response:
        return [
            {"role": "user", "content": str(instruction).strip()},
            {"role": "assistant", "content": str(response).strip()},
        ]
    return []


def _normalize_message(message: Dict[str, Any]) -> Dict[str, str]:
    return {
        "role": str(message.get("role", "")).lower().strip(),
        "content": str(message.get("content", "")).strip(),
    }


def _encode_messages(
    tokenizer,
    messages: List[Dict[str, str]],
    max_length: int,
    assistant_only_loss: bool,
    enable_thinking: Optional[bool],
) -> Dict[str, List[int]]:
    encoded_with_mask = _encode_with_template_mask(
        tokenizer,
        messages,
        max_length=max_length,
        assistant_only_loss=assistant_only_loss,
        enable_thinking=enable_thinking,
    )
    if encoded_with_mask is not None:
        return encoded_with_mask

    text = _render_chat(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    if not assistant_only_loss:
        labels = list(input_ids)
    else:
        labels = [-100] * len(input_ids)
        _mark_assistant_content_labels(tokenizer, messages, input_ids, labels)
        if not any(label != -100 for label in labels):
            raise ValueError(
                "Could not locate assistant response tokens for assistant-only SFT. "
                "Check the chat template or set assistant_only_loss=False."
            )

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _encode_with_template_mask(
    tokenizer,
    messages,
    max_length: int,
    assistant_only_loss: bool,
    enable_thinking: Optional[bool],
):
    if not assistant_only_loss:
        return None
    try:
        encoded = _render_chat(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_length,
        )
    except Exception:
        return None

    if not isinstance(encoded, dict):
        return None
    input_ids = _first_sequence(encoded.get("input_ids"))
    if input_ids is None:
        return None
    attention_mask = _first_sequence(encoded.get("attention_mask")) or [1] * len(input_ids)
    assistant_mask = None
    for key in ("assistant_masks", "assistant_tokens_mask", "assistant_mask"):
        if key in encoded:
            assistant_mask = _first_sequence(encoded[key])
            break
    if not assistant_mask or not any(assistant_mask):
        return None
    labels = [
        token_id if active else -100
        for token_id, active in zip(input_ids, assistant_mask)
    ]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _render_chat(tokenizer, messages, enable_thinking: Optional[bool], **kwargs):
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        call_kwargs = dict(kwargs)
        if enable_thinking is not None:
            call_kwargs["enable_thinking"] = enable_thinking
        try:
            return tokenizer.apply_chat_template(messages, **call_kwargs)
        except TypeError:
            call_kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **call_kwargs)

    if kwargs.get("tokenize"):
        rendered = _manual_chat_text(messages, kwargs.get("add_generation_prompt", False))
        return tokenizer(
            rendered,
            add_special_tokens=False,
            truncation=kwargs.get("truncation", False),
            max_length=kwargs.get("max_length"),
        )
    return _manual_chat_text(messages, kwargs.get("add_generation_prompt", False))


def _manual_chat_text(messages, add_generation_prompt=False):
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("assistant:")
    return "\n".join(parts)


def _mark_assistant_content_labels(tokenizer, messages, input_ids, labels):
    cursor = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        stripped = content.strip()
        candidates = [content, stripped, " " + stripped, "\n" + stripped]
        found = None
        found_ids = []
        for candidate in candidates:
            token_ids = tokenizer(candidate, add_special_tokens=False).get("input_ids", [])
            if not token_ids:
                continue
            start = _find_subsequence(input_ids, token_ids, cursor)
            if start is not None:
                found = start
                found_ids = token_ids
                break
        if found is None:
            continue
        end = min(found + len(found_ids), len(labels))
        for index in range(found, end):
            labels[index] = input_ids[index]
        cursor = end


def _find_subsequence(values: Sequence[int], pattern: Sequence[int], start: int = 0):
    if not pattern:
        return None
    max_start = len(values) - len(pattern)
    for index in range(start, max_start + 1):
        if list(values[index:index + len(pattern)]) == list(pattern):
            return index
    return None


def _first_sequence(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if not value:
        return []
    if isinstance(value[0], list):
        return list(value[0])
    return list(value)


def _apply_lora(model, r, alpha, dropout, target_modules):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "use_lora=True requires PEFT. Install it with: pip install peft"
        ) from exc

    modules = list(target_modules) if target_modules is not None else [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=modules,
    )
    return get_peft_model(model, config)


def _resolve_dtype(dtype: str):
    if dtype in (None, "auto"):
        return "auto"
    normalized = str(dtype).lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(
            "dtype must be one of: auto, float16, fp16, bfloat16, bf16, float32, fp32."
        )
    return mapping[normalized]


def _normalize_device_map(device_map):
    if device_map is None:
        return None
    if isinstance(device_map, str) and device_map.lower().strip() in {"", "none", "null"}:
        return None
    return device_map


def _ensure_padding_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id, eos_token, or unk_token.")


def _model_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Public aliases for advanced users who need to build custom Hugging Face SFT
# loops while reusing ArcLM's dataset, collator, dtype, device, and LoRA helpers.
HuggingFaceSFTDataset = _HuggingFaceSFTDataset
SFTDataCollator = _SFTDataCollator
apply_lora = _apply_lora
ensure_padding_token = _ensure_padding_token
load_sft_records = _load_sft_records
model_input_device = _model_input_device
normalize_device_map = _normalize_device_map
resolve_dtype = _resolve_dtype
set_seed = _set_seed

__all__ = [
    "SFTTrainingResult",
    "train_sft",
    "HuggingFaceSFTDataset",
    "SFTDataCollator",
    "apply_lora",
    "ensure_padding_token",
    "load_sft_records",
    "model_input_device",
    "normalize_device_map",
    "resolve_dtype",
    "set_seed",
]
