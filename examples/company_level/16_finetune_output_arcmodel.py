"""Download data, pre-fine-tune, then SFT-fine-tune an output `.arcmodel`."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Dataset, Model, Runtime, Trainer


PRETRAIN_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SFT_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


def download_to_text(url: str, path: Path, *, max_chars: int) -> Path:
    """Download a text dataset and save a bounded local copy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=60) as response:
        text = response.read(max_chars).decode("utf-8", errors="replace")
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def download_json(url: str, *, max_bytes: int) -> Any:
    with urlopen(url, timeout=120) as response:
        return json.loads(response.read(max_bytes).decode("utf-8", errors="replace"))


def prepare_pre_finetune_data(url: str, path: Path, *, max_chars: int) -> Path:
    """Prepare ArcLM plain-text data for continued language-model training."""

    return download_to_text(url, path, max_chars=max_chars)


def prepare_sft_data(url: str, path: Path, *, max_records: int) -> Path:
    """Prepare ArcLM JSONL records from Alpaca-style instruction data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = download_json(url, max_bytes=32 * 1024 * 1024)
    if not isinstance(rows, list):
        raise ValueError("SFT dataset must be a JSON list of instruction records.")

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if count >= max_records:
                break
            instruction = str(row.get("instruction") or "").strip()
            input_text = str(row.get("input") or "").strip()
            output = str(row.get("output") or "").strip()
            if not instruction or not output:
                continue
            prompt = format_instruction_prompt(instruction, input_text)
            handle.write(json.dumps({"prompt": prompt, "completion": " " + output}, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        raise ValueError("No usable SFT records were written.")
    return path


def format_instruction_prompt(instruction: str, input_text: str = "") -> str:
    if input_text:
        return f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
    return f"Instruction: {instruction}\nResponse:"


def train_stage(
    *,
    name: str,
    model: Model,
    dataset_path: Path,
    dataset_format: str,
    epochs: int,
    batch_size: int,
    block_size: int,
    learning_rate: float,
    debug: bool,
) -> dict[str, Any]:
    """Run one ArcLM full-fine-tune stage."""

    dataset = Dataset.load(dataset_path, format=dataset_format)
    train_block_size = min(int(block_size), int(model.config.block_size))
    print(f"\n{name}")
    print(f"Data: {dataset_path}")
    print(f"Format: {dataset_format}")
    print(f"Block size: {train_block_size}")

    trainer = Trainer(
        model=model,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        block_size=train_block_size,
    )
    history = trainer.train(mode="full_finetune", debug=debug)
    print(f"{name} losses: {history['train_losses']}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-fine-tune and SFT-fine-tune an ArcLM model from output/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Full default run:
    python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel

  Quick smoke run:
    python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel --pretrain-max-chars 512 --sft-max-records 1 --batch-size 64 --block-size 16 --max-new-tokens 3

  Only pre-fine-tuning:
    python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel --skip-sft

  Only SFT fine-tuning:
    python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel --skip-pretrain

  Use custom dataset URLs:
    python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel --pretrain-url https://example.com/train.txt --sft-url https://example.com/instructions.json
""",
    )
    parser.add_argument("model", nargs="?", default="output/MiniGPT.arcmodel", help="Legacy or native .arcmodel to fine-tune.")
    parser.add_argument("--pretrain-url", default=PRETRAIN_URL, help="Plain-text dataset URL for pre-fine-tuning.")
    parser.add_argument("--sft-url", default=SFT_URL, help="Alpaca-style JSON dataset URL for SFT.")
    parser.add_argument("--pretrain-data", default="data/downloaded_pre_finetune.txt")
    parser.add_argument("--sft-data", default="data/downloaded_sft.jsonl")
    parser.add_argument("--save-pretrained-to", default="output/MiniGPT-pre-finetuned.arcmodel")
    parser.add_argument("--save-sft-to", default="output/MiniGPT-sft.arcmodel")
    parser.add_argument("--prompt", default="Instruction: Describe a time when you had to make a difficult decision.\nResponse:")
    parser.add_argument("--pretrain-max-chars", type=int, default=16_000)
    parser.add_argument("--sft-max-records", type=int, default=8)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--sft-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--pretrain-learning-rate", type=float, default=5e-5)
    parser.add_argument("--sft-learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--debug", action="store_true", help="Print every training step loss.")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--skip-sft", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = Runtime.auto(prefer="cuda")

    model = Model.load(args.model, runtime=runtime)
    print(f"Loaded: {args.model}")
    print(f"Architecture: {model.inspect()['architecture_id']}")
    print(f"Tokenizer: {type(model.tokenizer).__name__}")
    print("Before:", model.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=0.0))

    if not args.skip_pretrain:
        pretrain_path = prepare_pre_finetune_data(
            args.pretrain_url,
            Path(args.pretrain_data),
            max_chars=args.pretrain_max_chars,
        )
        train_stage(
            name="Pre-fine-tuning",
            model=model,
            dataset_path=pretrain_path,
            dataset_format="txt",
            epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            block_size=args.block_size,
            learning_rate=args.pretrain_learning_rate,
            debug=args.debug,
        )
        model.save(args.save_pretrained_to, overwrite=True)
        print(f"Saved pre-fine-tuned native artifact: {args.save_pretrained_to}")

    if not args.skip_sft:
        sft_path = prepare_sft_data(args.sft_url, Path(args.sft_data), max_records=args.sft_max_records)
        train_stage(
            name="SFT fine-tuning",
            model=model,
            dataset_path=sft_path,
            dataset_format="jsonl",
            epochs=args.sft_epochs,
            batch_size=args.batch_size,
            block_size=args.block_size,
            learning_rate=args.sft_learning_rate,
            debug=args.debug,
        )
        model.save(args.save_sft_to, overwrite=True)
        reloaded = Model.load(args.save_sft_to, runtime=runtime)
        print(f"Saved SFT native artifact: {args.save_sft_to}")
        print("After:", reloaded.generate(args.prompt, max_new_tokens=args.max_new_tokens, temperature=0.0))


if __name__ == "__main__":
    main()
