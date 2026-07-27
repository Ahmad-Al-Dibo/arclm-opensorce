"""Hugging Face SFT: run a one-step smoke test with a tiny model."""

from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import train_sft


def main():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install Hugging Face dependencies with: pip install -e .[hf]") from exc

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset = root / "sft.jsonl"
        output = root / "hf_sft"
        dataset.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "What is ArcLM?"},
                        {"role": "assistant", "content": "ArcLM is a compact training toolkit."},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = train_sft(
            model="hf-internal-testing/tiny-random-gpt2",
            dataset=str(dataset),
            output_dir=str(output),
            backend="huggingface",
            assistant_only_loss=True,
            use_lora=False,
            batch_size=1,
            learning_rate=5e-5,
            num_epochs=1,
            max_steps=1,
            max_length=128,
            trust_remote_code=False,
        )

        print(f"SFT steps: {result.steps}")


if __name__ == "__main__":
    main()
