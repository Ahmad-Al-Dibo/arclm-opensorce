"""Fine-tuning: adapt a native `.arcmodel` with the new Trainer API."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Dataset, Model, Runtime, Trainer


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base_data = root / "base.txt"
        task_data = root / "task.txt"
        base_artifact = root / "base.arcmodel"
        tuned_artifact = root / "tuned.arcmodel"

        base_data.write_text("ArcLM learns compact training examples. " * 40, encoding="utf-8")
        task_data.write_text("Question: What is ArcLM?\nAnswer: ArcLM is a compact toolkit.\n" * 20, encoding="utf-8")

        runtime = Runtime.auto(prefer="cpu")
        base_dataset = Dataset.load(base_data)
        prepared = base_dataset.prepare(block_size=8, batch_size=2)
        model = Model.create(
            architecture="arclm-native",
            tokenizer=prepared.tokenizer,
            runtime=runtime,
            embed_dim=16,
            block_size=8,
            num_blocks=1,
            dropout=0.0,
        )
        Trainer(model=model, dataset=base_dataset, epochs=1, batch_size=2, learning_rate=1e-3, block_size=8).train()
        model.save(base_artifact, overwrite=True)

        tuned = Model.load(base_artifact, runtime=runtime)
        history = Trainer(
            model=tuned,
            dataset=Dataset.load(task_data),
            epochs=1,
            batch_size=2,
            learning_rate=5e-4,
            block_size=8,
        ).train(mode="full_finetune")
        artifact = tuned.save(tuned_artifact, overwrite=True)

        print(f"Fine-tune losses: {history['train_losses']}")
        print(f"Fine-tuned ArcLM artifact: {artifact.path}")


if __name__ == "__main__":
    main()
