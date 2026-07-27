"""Quickstart: train a tiny native ArcLM checkpoint from generated text."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import train_model


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "demo.txt"
        model_path = root / "demo_arclm.pth"

        data_path.write_text(
            "ArcLM trains compact language models. "
            "Small examples make the training loop easy to inspect. " * 24,
            encoding="utf-8",
        )

        result = train_model(
            mode="pretrain",
            data=str(data_path),
            output=str(model_path),
            tokenizer_type="word",
            max_vocab=200,
            embed_dim=32,
            num_blocks=1,
            block_size=16,
            batch_size=4,
            num_epochs=1,
            learning_rate=1e-3,
            validation_split=0.0,
            training_log_interval=0,
            device="cpu",
        )

        print(f"Saved checkpoint: {result.model_path}")
        print(f"Vocabulary size: {result.vocab_size}")


if __name__ == "__main__":
    main()
