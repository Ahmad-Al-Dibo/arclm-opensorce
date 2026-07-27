"""ArcLM SFT-pijplijn voor Qwen op FreedomIntelligence medical-o1 data.

Dit script gebruikt ArcLM 0.4.0's Hugging Face SFT-formaat:
JSONL regels met een "messages" lijst van chatberichten. ArcLM's
"train_sft" helper slaat het uiteindelijke adapter/model op; dit script
volgt dezelfde backend-stroom maar voegt periodieke checkpoints toe voor
langere runs.
Voorbeeld gebruik:
    python train.py \
        --dataset-name FreedomIntelligence/medical-o1-reasoning-SFT \
        --dataset-config en \
        --model Qwen/Qwen3-0.6B \
        --output-dir models/qwen_medical_o1_lora \
        --batch-size 1 \
        --gradient-accumulation-steps 8 \
        --learning-rate 2e-4 \
        --num-epochs 1 \
        --max-length 2048 \
        --lora-r 16 \
        --lora-alpha 32 \
        --lora-dropout 0.05 \
        --lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj, up_proj,down_proj \
        --checkpoint-steps 100 \
        --save-epoch-checkpoints \
        --enable-thinking \
        --trust-remote-code

Wil je checkpoints opslaan: gebruik --checkpoint-steps N om elke N stappen op te slaan, of --save-epoch-checkpoints om aan het eind van elke epoch op te slaan. Gebruik beide als je beide wilt.
Het script splitst de dataset in train/validation/test, kijkt naar tokenizer-lengtes en draait daarna de SFT-trainingsloop met optionele LoRA-adapters. Het eindmodel en metadata worden in de opgegeven output-map opgeslagen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from datasets import Dataset, get_dataset_config_names, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from arclm import DataProcessor, SFTTrainingResult, get_version
from arclm.sft import (
    HuggingFaceSFTDataset,
    SFTDataCollator,
    apply_lora,
    ensure_padding_token,
    load_sft_records,
    model_input_device,
    normalize_device_map,
    resolve_dtype,
    set_seed,
)


DEFAULT_DATASET = "FreedomIntelligence/medical-o1-reasoning-SFT"
DEFAULT_DATASET_CONFIG = "en"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_DIR = Path("models") / "qwen_medical_o1_lora"
DEFAULT_DATA_DIR = Path("data") / "medical_o1_reasoning_sft"
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful medical question-answering assistant. Answer accurately "
    "from the provided question. This model is for research and must not replace "
    "professional medical advice."
)
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Qwen model with ArcLM on medical-o1-reasoning-SFT."
    )

    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset-name", default=DEFAULT_DATASET)
    data.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    data.add_argument("--dataset-split", default="train")
    data.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    data.add_argument("--validation-size", type=float, default=0.05)
    data.add_argument("--test-size", type=float, default=0.05)
    data.add_argument("--max-samples", type=int, default=None)
    data.add_argument(
        "--answer-format",
        choices=["reasoning_and_answer", "answer_only"],
        default="reasoning_and_answer",
        help="Use Complex_CoT + Response, or train only on Response.",
    )
    data.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    data.add_argument("--overwrite-data", action="store_true")

    model = parser.add_argument_group("model and tokenizer")
    model.add_argument("--model", default=DEFAULT_MODEL)
    model.add_argument("--trust-remote-code", action="store_true", default=True)
    model.add_argument("--enable-thinking", action="store_true")
    model.add_argument("--max-length", type=int, default=2048)
    model.add_argument("--dtype", default="auto")
    model.add_argument("--device-map", default="auto")
    model.add_argument("--token-stats-samples", type=int, default=1000)
    model.add_argument("--skip-tokenizer-analysis", action="store_true")

    training = parser.add_argument_group("training")
    training.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    training.add_argument("--batch-size", type=int, default=1)
    training.add_argument("--gradient-accumulation-steps", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=2e-4)
    training.add_argument("--num-epochs", type=int, default=1)
    training.add_argument("--max-steps", type=int, default=None)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--prepare-only", action="store_true")
    training.add_argument("--gradient-checkpointing", action="store_true")

    lora = parser.add_argument_group("LoRA")
    lora.add_argument("--no-lora", action="store_true")
    lora.add_argument("--lora-r", type=int, default=16)
    lora.add_argument("--lora-alpha", type=int, default=32)
    lora.add_argument("--lora-dropout", type=float, default=0.05)
    lora.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_LORA_TARGETS),
        help="Comma-separated target module names.",
    )

    checkpointing = parser.add_argument_group("checkpointing")
    checkpointing.add_argument(
        "--checkpoint-steps",
        type=int,
        default=100,
        help="Save a checkpoint every N optimizer steps. Use 0 to disable.",
    )
    checkpointing.add_argument("--checkpoint-dir", type=Path, default=None)
    checkpointing.add_argument("--save-epoch-checkpoints", action="store_true")
    checkpointing.add_argument("--no-save-optimizer-state", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(f"ArcLM version: {get_version()}")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Qwen SFT on CPU is only practical for smoke tests.")

    prepared = prepare_dataset(args)
    tokenizer_stats = None
    if not args.skip_tokenizer_analysis:
        tokenizer = load_qwen_tokenizer(args)
        tokenizer_stats = analyze_token_lengths(
            prepared["files"]["train"],
            tokenizer=tokenizer,
            max_records=args.token_stats_samples,
            max_length=args.max_length,
            enable_thinking=args.enable_thinking,
        )
        print_json("Tokenizer length analysis", tokenizer_stats)
        # tokenizer.save_pretrained(args.output_dir / "tokenizer")

    run_config = build_run_config(args, prepared, tokenizer_stats)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_config.json", run_config)

    if args.prepare_only:
        print("Prepared data only. Training was not started.")
        return

    result = train_with_arclm_sft_checkpoints(args, prepared["files"]["train"])
    print_json("Training result", asdict(result))


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be greater than zero.")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be greater than zero.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than zero.")
    if args.validation_size < 0 or args.test_size < 0:
        raise ValueError("Split sizes must be non-negative.")
    if args.validation_size + args.test_size >= 1:
        raise ValueError("--validation-size + --test-size must be less than 1.")


def prepare_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.data_dir / f"{args.dataset_config}_train.jsonl"
    validation_path = args.data_dir / f"{args.dataset_config}_validation.jsonl"
    test_path = args.data_dir / f"{args.dataset_config}_test.jsonl"
    metadata_path = args.data_dir / f"{args.dataset_config}_metadata.json"

    if (
        train_path.exists()
        and validation_path.exists()
        and test_path.exists()
        and metadata_path.exists()
        and not args.overwrite_data
    ):
        metadata = read_json(metadata_path)
        metadata["files"] = {
            "train": train_path,
            "validation": validation_path,
            "test": test_path,
            "metadata": metadata_path,
        }
        print(f"Using prepared dataset files from {args.data_dir}")
        return metadata

    available_configs = get_dataset_config_names(args.dataset_name)
    if args.dataset_config not in available_configs:
        raise ValueError(
            f"Dataset config '{args.dataset_config}' is not available. "
            f"Choose one of: {available_configs}"
        )

    raw = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    raw = raw.filter(has_required_fields)
    if args.max_samples is not None:
        raw = raw.select(range(min(args.max_samples, len(raw))))

    analysis = analyze_dataset(raw)
    print_json("Dataset structure", analysis)

    splits = split_dataset(
        raw,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    counts = {
        "train": write_arc_chat_jsonl(splits["train"], train_path, args),
        "validation": write_arc_chat_jsonl(splits["validation"], validation_path, args),
        "test": write_arc_chat_jsonl(splits["test"], test_path, args),
    }
    arclm_verification = verify_arc_dataset(train_path)

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "available_configs": available_configs,
        "answer_format": args.answer_format,
        "system_prompt": args.system_prompt,
        "raw_rows_after_filtering": len(raw),
        "prepared_rows": counts,
        "analysis": analysis,
        "arclm_verification": arclm_verification,
        "files": {
            "train": train_path,
            "validation": validation_path,
            "test": test_path,
            "metadata": metadata_path,
        },
    }
    write_json(metadata_path, serializable(metadata))
    return metadata


def has_required_fields(row: Dict[str, Any]) -> bool:
    return bool(str(row.get("Question", "")).strip()) and bool(
        str(row.get("Response", "")).strip()
    )


def analyze_dataset(dataset: Dataset) -> Dict[str, Any]:
    columns = list(dataset.column_names)
    blanks = {column: 0 for column in columns}
    lengths: Dict[str, List[int]] = {column: [] for column in columns}

    for row in dataset:
        for column in columns:
            value = row.get(column)
            text = "" if value is None else str(value)
            if not text.strip():
                blanks[column] += 1
            lengths[column].append(len(text))

    return {
        "rows": len(dataset),
        "columns": columns,
        "features": {name: str(feature) for name, feature in dataset.features.items()},
        "blank_values": blanks,
        "character_lengths": {
            column: summarize_numbers(values) for column, values in lengths.items()
        },
        "recommended_mapping": {
            "Question": "user message",
            "Complex_CoT": "assistant reasoning when answer_format=reasoning_and_answer",
            "Response": "assistant final answer",
        },
    }


def summarize_numbers(values: Sequence[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": round(statistics.mean(ordered), 2),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def percentile(ordered_values: Sequence[int], q: float) -> int:
    index = min(len(ordered_values) - 1, max(0, math.ceil(len(ordered_values) * q) - 1))
    return int(ordered_values[index])


def split_dataset(
    dataset: Dataset,
    validation_size: float,
    test_size: float,
    seed: int,
) -> Dict[str, Dataset]:
    total = len(dataset)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)

    validation_count = split_count(total, validation_size)
    test_count = split_count(total, test_size)
    if validation_count + test_count >= total:
        overflow = validation_count + test_count - max(0, total - 1)
        test_count = max(0, test_count - overflow)

    validation_indices = indices[:validation_count]
    test_indices = indices[validation_count:validation_count + test_count]
    train_indices = indices[validation_count + test_count:]

    return {
        "train": dataset.select(train_indices),
        "validation": dataset.select(validation_indices),
        "test": dataset.select(test_indices),
    }


def split_count(total: int, fraction: float) -> int:
    if fraction <= 0 or total <= 1:
        return 0
    count = int(round(total * fraction))
    return max(1, count)


def write_arc_chat_jsonl(dataset: Dataset, path: Path, args: argparse.Namespace) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            json.dump(row_to_arc_messages(row, args), handle, ensure_ascii=False)
            handle.write("\n")
            count += 1
    return count


def row_to_arc_messages(row: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    question = normalize_text(row["Question"])
    response = normalize_text(row["Response"])
    cot = normalize_text(row.get("Complex_CoT", ""))

    if args.answer_format == "reasoning_and_answer" and cot:
        assistant = f"Reasoning:\n{cot}\n\nFinal answer:\n{response}"
    else:
        assistant = response

    messages = []
    if args.system_prompt.strip():
        messages.append({"role": "system", "content": args.system_prompt.strip()})
    messages.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
        ]
    )
    return {"messages": messages}


def normalize_text(value: Any) -> str:
    lines = str(value).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.strip() for line in lines).strip()


def verify_arc_dataset(train_path: Path) -> Dict[str, Any]:
    loaded = DataProcessor.load(train_path).clean()
    records = load_sft_records(str(train_path))
    roles = sorted({message["role"] for record in records for message in record})
    return {"rows_loaded_by_arclm": len(loaded.samples), "sft_records": len(records), "roles": roles}


def load_qwen_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_padding_token(tokenizer)
    return tokenizer


def analyze_token_lengths(
    jsonl_path: Path,
    tokenizer: Any,
    max_records: int,
    max_length: int,
    enable_thinking: bool,
) -> Dict[str, Any]:
    lengths: List[int] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= max_records:
                break
            row = json.loads(line)
            token_ids = render_chat_ids(tokenizer, row["messages"], enable_thinking=enable_thinking)
            lengths.append(len(token_ids))

    return {
        "records_sampled": len(lengths),
        "chat_token_lengths": summarize_numbers(lengths),
        "max_length": max_length,
        "records_over_max_length": sum(1 for length in lengths if length > max_length),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def render_chat_ids(tokenizer: Any, messages: List[Dict[str, str]], enable_thinking: bool) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        kwargs = {"tokenize": True, "add_generation_prompt": False}
        try:
            return flatten_token_ids(
                tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=enable_thinking,
                    **kwargs,
                )
            )
        except TypeError:
            return flatten_token_ids(tokenizer.apply_chat_template(messages, **kwargs))

    text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def flatten_token_ids(value: Any) -> List[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids", [])
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        return list(value[0])
    return list(value)


def train_with_arclm_sft_checkpoints(args: argparse.Namespace, train_path: Path) -> SFTTrainingResult:
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or (output_dir / "checkpoints")
    use_lora = not args.no_lora

    tokenizer = load_qwen_tokenizer(args)
    model = load_causal_lm(args)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if use_lora:
        model = apply_lora(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=parse_lora_targets(args.lora_target_modules),
        )
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    records = load_sft_records(str(train_path))
    train_dataset = HuggingFaceSFTDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        assistant_only_loss=True,
        enable_thinking=args.enable_thinking,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SFTDataCollator(tokenizer),
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for SFT.")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    loss_history = run_training_loop(
        model=model,
        loader=loader,
        optimizer=optimizer,
        tokenizer=tokenizer,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        args=args,
    )

    save_hf_checkpoint(
        model=model,
        tokenizer=tokenizer,
        directory=output_dir,
        optimizer=optimizer,
        training_state={
            "kind": "final",
            "steps": len(loss_history),
            "loss_history": loss_history,
        },
        save_optimizer_state=not args.no_save_optimizer_state,
    )

    metadata_path = output_dir / "arclm_sft_metadata.json"
    result = SFTTrainingResult(
        model=args.model,
        dataset=str(train_path),
        output_dir=str(output_dir),
        backend="huggingface",
        use_lora=use_lora,
        assistant_only_loss=True,
        train_loss_history=loss_history,
        steps=len(loss_history),
        metadata_path=str(metadata_path),
        adapter_path=str(output_dir) if use_lora else None,
        full_model_path=None if use_lora else str(output_dir),
    )
    write_json(metadata_path, asdict(result))
    return result


def load_causal_lm(args: argparse.Namespace):
    model_kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    resolved_dtype = resolve_dtype(args.dtype)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    device_map = normalize_device_map(args.device_map)
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except TypeError:
        if "torch_dtype" in model_kwargs:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    if device_map is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    return model


def run_training_loop(
    model: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    output_dir: Path,
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> List[float]:
    model.train()
    loss_history: List[float] = []
    completed_steps = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.num_epochs):
        for batch_index, batch in enumerate(loader, start=1):
            device = model_input_device(model)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / args.gradient_accumulation_steps
            loss.backward()

            should_step = batch_index % args.gradient_accumulation_steps == 0
            should_step = should_step or batch_index == len(loader)
            if not should_step:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1
            current_loss = float(raw_loss.detach().cpu().item())
            loss_history.append(current_loss)
            print(
                f"SFT epoch {epoch + 1}/{args.num_epochs} | "
                f"step {completed_steps} | loss {current_loss:.4f}",
                flush=True,
            )

            if args.checkpoint_steps and completed_steps % args.checkpoint_steps == 0:
                save_hf_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    directory=checkpoint_dir / f"step-{completed_steps:06d}",
                    optimizer=optimizer,
                    training_state={
                        "kind": "step",
                        "epoch": epoch + 1,
                        "step": completed_steps,
                        "loss": current_loss,
                        "output_dir": str(output_dir),
                    },
                    save_optimizer_state=not args.no_save_optimizer_state,
                )

            if args.max_steps is not None and completed_steps >= args.max_steps:
                return loss_history

        if args.save_epoch_checkpoints:
            save_hf_checkpoint(
                model=model,
                tokenizer=tokenizer,
                directory=checkpoint_dir / f"epoch-{epoch + 1:03d}",
                optimizer=optimizer,
                training_state={
                    "kind": "epoch",
                    "epoch": epoch + 1,
                    "step": completed_steps,
                    "loss": loss_history[-1] if loss_history else None,
                    "output_dir": str(output_dir),
                },
                save_optimizer_state=not args.no_save_optimizer_state,
            )

    return loss_history


def save_hf_checkpoint(
    model: Any,
    tokenizer: Any,
    directory: Path,
    optimizer: Optional[torch.optim.Optimizer],
    training_state: Dict[str, Any],
    save_optimizer_state: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    write_json(directory / "arclm_checkpoint_metadata.json", training_state)
    if save_optimizer_state and optimizer is not None:
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "training_state": training_state,
            },
            directory / "training_state.pt",
        )
    print(f"Saved checkpoint: {directory}", flush=True)


def parse_lora_targets(value: str) -> Optional[List[str]]:
    targets = [part.strip() for part in value.split(",") if part.strip()]
    return targets or None


def build_run_config(
    args: argparse.Namespace,
    prepared: Dict[str, Any],
    tokenizer_stats: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "arclm_version": get_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "args": serializable(vars(args)),
        "prepared_dataset": serializable(prepared),
        "tokenizer_stats": tokenizer_stats,
        "training_parameters": {
            "model": args.model,
            "backend": "huggingface",
            "assistant_only_loss": True,
            "use_lora": not args.no_lora,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "device_map": args.device_map,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": parse_lora_targets(args.lora_target_modules),
            "checkpoint_steps": args.checkpoint_steps,
        },
    }


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), indent=2, ensure_ascii=False), encoding="utf-8")


def print_json(title: str, value: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print(json.dumps(serializable(value), indent=2, ensure_ascii=False))


def serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
