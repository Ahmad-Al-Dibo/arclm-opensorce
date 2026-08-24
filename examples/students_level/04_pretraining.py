"""Pretraining: use Dataset, Model, Trainer, and Runtime directly."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Dataset, Model, Runtime, Trainer


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "pretrain.txt"
        artifact_path = root / "pretrained.arcmodel"
        data_path.write_text(
            "Language models learn token patterns from text. "
            "ArcLM keeps each training step inspectable. " * 32,
            encoding="utf-8",
        )

        runtime = Runtime.auto(prefer="cpu")
        dataset = Dataset.load(data_path)
        prepared = dataset.prepare(block_size=8, batch_size=2)
        model = Model.create(
            architecture="arclm-native",
            tokenizer=prepared.tokenizer,
            runtime=runtime,
            embed_dim=16,
            block_size=8,
            num_blocks=1,
            dropout=0.0,
        )
        trainer = Trainer(
            model=model,
            dataset=dataset,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            block_size=8,
        )

        history = trainer.train(mode="pretrain")
        artifact = model.save(artifact_path, overwrite=True)

        print(f"Losses: {history['train_losses']}")
        print(f"Saved ArcLM artifact: {artifact.path}")


if __name__ == "__main__":
    main()
