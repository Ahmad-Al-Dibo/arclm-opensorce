"""Generate Qwen3-0.6B answers after ArcLM SFT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from test_base_model import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    generate_for_prompts,
    load_hf_model,
    write_jsonl,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER = EXAMPLE_DIR / "output" / "qwen3_0_6b_sft_lora"
DEFAULT_OUTPUT = EXAMPLE_DIR / "output" / "finetuned_outputs.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Test the fine-tuned Qwen3-0.6B model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    tokenizer, model = load_finetuned_model(
        model_name=args.model,
        adapter_or_model_dir=Path(args.adapter),
        dtype=args.dtype,
        device_map=args.device_map,
    )
    rows = generate_for_prompts(
        tokenizer=tokenizer,
        model=model,
        prompts_path=Path(args.prompts),
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(Path(args.output), rows)


def load_finetuned_model(model_name: str, adapter_or_model_dir: Path, dtype: str, device_map: str):
    if not adapter_or_model_dir.exists():
        raise SystemExit(
            f"Fine-tuned output not found: {adapter_or_model_dir}\n"
            "Run train_qwen3_0_6b_sft.py first."
        )

    if (adapter_or_model_dir / "adapter_config.json").exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Loading a LoRA adapter requires PEFT. Install with: pip install peft") from exc

        tokenizer, base_model = load_hf_model(model_name, dtype=dtype, device_map=device_map)
        model = PeftModel.from_pretrained(base_model, adapter_or_model_dir)
        model.eval()
        return tokenizer, model

    return load_hf_model(str(adapter_or_model_dir), dtype=dtype, device_map=device_map)


if __name__ == "__main__":
    main()
