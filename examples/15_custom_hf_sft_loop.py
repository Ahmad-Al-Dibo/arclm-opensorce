"""Custom Hugging Face SFT loop using ArcLM's public SFT building blocks.

This example is for advanced users who want the lower-level pieces behind
``train_sft()`` while still using ArcLM's dataset parsing, collator, dtype,
device, seed, and optional LoRA helpers.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run a custom HF SFT loop with ArcLM helpers.")
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-gpt2")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device-map", default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "examples" / "_outputs" / "custom_hf_sft_loop"))
    return parser.parse_args()


def write_demo_dataset(path: Path) -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": "What is ArcLM?"},
                {
                    "role": "assistant",
                    "content": "ArcLM is a compact Python toolkit for language-model training.",
                },
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What does SFT do?"},
                {
                    "role": "assistant",
                    "content": "SFT teaches a model from examples of desired responses.",
                },
            ]
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def load_model_and_tokenizer(model_name: str, dtype: str, device_map_value: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install Hugging Face dependencies with: pip install -e .[hf]") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    ensure_padding_token(tokenizer)

    device_map = normalize_device_map(device_map_value)
    model_kwargs = {"trust_remote_code": False}
    resolved_dtype = resolve_dtype(dtype)
    if resolved_dtype is not None:
        model_kwargs["dtype"] = resolved_dtype
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device_map is None:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str) -> str:
    model.eval()
    device = model_input_device(model)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def train_one_loop(model, loader, learning_rate: float, max_steps: int):
    model.train()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
    )
    loss_history = []
    completed_steps = 0

    for batch in loader:
        device = model_input_device(model)
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        completed_steps += 1
        loss_history.append(float(loss.detach().cpu().item()))
        print(f"step {completed_steps} | loss {loss_history[-1]:.4f}")
        if completed_steps >= max_steps:
            break

    return loss_history


def run(args) -> SFTTrainingResult:
    set_seed(args.seed)
    print(f"ArcLM version: {get_version()}")

    output_dir = Path(args.output_dir)
    root = output_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "custom_hf_sft.jsonl"
    write_demo_dataset(dataset_path)

    processed = DataProcessor.load(dataset_path).clean()
    records = load_sft_records(str(dataset_path))
    print(f"loaded rows: {len(processed.samples)} | SFT records: {len(records)}")

    tokenizer, model = load_model_and_tokenizer(args.model, args.dtype, args.device_map)
    print("before:", generate_text(tokenizer, model, "ArcLM is"))

    if args.use_lora:
        model = apply_lora(
            model,
            r=4,
            alpha=8,
            dropout=0.0,
            target_modules=["c_attn", "c_proj"],
        )

    train_dataset = HuggingFaceSFTDataset(
        records=records,
        tokenizer=tokenizer,
        max_length=args.max_length,
        assistant_only_loss=True,
        enable_thinking=False,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=SFTDataCollator(tokenizer),
    )

    loss_history = train_one_loop(model, loader, args.learning_rate, args.max_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    result = SFTTrainingResult(
        model=args.model,
        dataset=str(dataset_path),
        output_dir=str(output_dir),
        backend="huggingface-custom-loop",
        use_lora=args.use_lora,
        assistant_only_loss=True,
        train_loss_history=loss_history,
        steps=len(loss_history),
        metadata_path=str(output_dir / "arclm_sft_metadata.json"),
        adapter_path=str(output_dir) if args.use_lora else None,
        full_model_path=None if args.use_lora else str(output_dir),
    )
    Path(result.metadata_path).write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print("after:", generate_text(tokenizer, model, "ArcLM is"))
    print(f"saved: {result.output_dir}")
    return result


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
