"""LoRA SFT: run a one-step PEFT smoke test with a tiny Hugging Face model."""

from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import train_sft


def main():
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install LoRA dependencies with: pip install -e .[hf]") from exc

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = root / "sft.jsonl"
        output = root / "lora_sft"
        dataset.write_text(
            json.dumps({"instruction": "Explain LoRA.", "output": "LoRA trains small adapter matrices."}) + "\n",
            encoding="utf-8",
        )

        result = train_sft(
            model="hf-internal-testing/tiny-random-gpt2",
            dataset=str(dataset),
            output_dir=str(output),
            backend="huggingface",
            assistant_only_loss=True,
            use_lora=True,
            lora_target_modules=["c_attn", "c_proj"],
            batch_size=1,
            learning_rate=1e-4,
            num_epochs=1,
            max_steps=1,
            max_length=128,
            trust_remote_code=False,
        )

        print(f"Adapter path: {result.adapter_path}")


if __name__ == "__main__":
    main()
