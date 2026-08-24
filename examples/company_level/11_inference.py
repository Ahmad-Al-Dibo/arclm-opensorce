"""Inference: train, save, load, and generate with the new Model API."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Dataset, Model, Runtime, Trainer


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "data.txt"
        artifact_path = root / "model.arcmodel"
        data_path.write_text("ArcLM generates text from compact native artifacts. " * 48, encoding="utf-8")

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
        Trainer(model=model, dataset=dataset, epochs=1, batch_size=2, learning_rate=1e-3, block_size=8).train()
        model.save(artifact_path, overwrite=True)

        loaded = Model.load(artifact_path, runtime=runtime)
        print(loaded.generate("ArcLM", max_new_tokens=8, temperature=0.0))


if __name__ == "__main__":
    main()
