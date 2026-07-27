"""Continued training: extend a compatible native checkpoint."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import train_model


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base_data = root / "base.txt"
        more_data = root / "more.txt"
        base_model = root / "base.pth"
        continued_model = root / "continued.pth"

        base_data.write_text("ArcLM starts from a small corpus. " * 40, encoding="utf-8")
        more_data.write_text("Continued training adds domain text. " * 40, encoding="utf-8")

        train_model(
            mode="pretrain",
            data=str(base_data),
            output=str(base_model),
            tokenizer_type="word",
            max_vocab=200,
            embed_dim=32,
            num_blocks=1,
            block_size=16,
            batch_size=4,
            num_epochs=1,
            validation_split=0.0,
            training_log_interval=0,
            device="cpu",
        )

        result = train_model(
            mode="continue_training",
            checkpoint=str(base_model),
            data=str(more_data),
            output=str(continued_model),
            embed_dim=32,
            num_blocks=1,
            block_size=16,
            batch_size=4,
            num_epochs=2,
            validation_split=0.0,
            training_log_interval=0,
            device="cpu",
        )

        print(f"Continued checkpoint: {result.model_path}")


if __name__ == "__main__":
    main()
