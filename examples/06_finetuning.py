"""Fine-tuning: adapt a native checkpoint with next-token loss."""

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
        task_data = root / "task.txt"
        base_model = root / "base.pth"
        tuned_model = root / "tuned.pth"

        base_data.write_text("ArcLM learns compact training examples. " * 40, encoding="utf-8")
        task_data.write_text("Question: What is ArcLM?\nAnswer: ArcLM is a compact toolkit.\n" * 20, encoding="utf-8")

        train_model(
            mode="pretrain",
            data=str(base_data),
            output=str(base_model),
            tokenizer_type="word",
            max_vocab=300,
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
            mode="finetune",
            checkpoint=str(base_model),
            data=str(task_data),
            output=str(tuned_model),
            embed_dim=32,
            num_blocks=1,
            block_size=16,
            batch_size=4,
            num_epochs=1,
            learning_rate=5e-4,
            validation_split=0.0,
            training_log_interval=0,
            device="cpu",
        )

        print(f"Fine-tuned checkpoint: {result.model_path}")


if __name__ == "__main__":
    main()
