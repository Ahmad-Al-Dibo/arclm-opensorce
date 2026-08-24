"""Continued training: load a native `.arcmodel` and keep training."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Dataset, Model, Runtime, Trainer


def _train(model: Model, dataset: Dataset, *, epochs: int = 1) -> dict:
    trainer = Trainer(
        model=model,
        dataset=dataset,
        epochs=epochs,
        batch_size=2,
        learning_rate=1e-3,
        block_size=8,
    )
    return trainer.train(mode="pretrain")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base_data = root / "base.txt"
        more_data = root / "more.txt"
        base_artifact = root / "base.arcmodel"
        continued_artifact = root / "continued.arcmodel"

        base_data.write_text("ArcLM starts from a small corpus. " * 40, encoding="utf-8")
        more_data.write_text("Continued training adds domain text. " * 40, encoding="utf-8")

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
        _train(model, base_dataset)
        model.save(base_artifact, overwrite=True)

        continued = Model.load(base_artifact, runtime=runtime)
        history = _train(continued, Dataset.load(more_data), epochs=2)
        artifact = continued.save(continued_artifact, overwrite=True)

        print(f"Continued losses: {history['train_losses']}")
        print(f"Continued ArcLM artifact: {artifact.path}")


if __name__ == "__main__":
    main()
