"""Inference: train a tiny checkpoint, load it, and generate text."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import load_model, train_model


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = root / "data.txt"
        model_path = root / "model.pth"
        data.write_text("ArcLM generates text from compact checkpoints. " * 48, encoding="utf-8")

        train_model(
            mode="pretrain",
            data=str(data),
            output=str(model_path),
            tokenizer_type="word",
            max_vocab=120,
            embed_dim=32,
            num_blocks=1,
            block_size=16,
            batch_size=4,
            num_epochs=1,
            validation_split=0.0,
            training_log_interval=0,
            device="cpu",
        )

        loaded = load_model(model_path, device="cpu")
        print(loaded.predict("ArcLM", max_new_tokens=8, top_k=5))


if __name__ == "__main__":
    main()
