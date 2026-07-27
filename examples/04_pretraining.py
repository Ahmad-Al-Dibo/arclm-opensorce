"""Pretraining: train a native ArcLM model from plain text."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import train_model


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "pretrain.txt"
        output = root / "arclm_pretrained.pth"
        data.write_text(
            "Language models learn token patterns from text. "
            "ArcLM keeps each training step inspectable. " * 32,
            encoding="utf-8",
        )

        result = train_model(
            mode="pretrain",
            data=str(data),
            output=str(output),
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

        print(result.mode)
        print(result.model_path)


if __name__ == "__main__":
    main()
