"""Quickstart: train and save a tiny ArcLM model with the Lab API."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Lab, Runtime


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data_path = root / "demo.txt"
        artifact_path = root / "demo.arcmodel"

        data_path.write_text(
            "ArcLM trains compact language models. "
            "Small examples make the training loop easy to inspect. " * 24,
            encoding="utf-8",
        )

        lab = Lab(runtime=Runtime.auto(prefer="cpu"))
        model = lab.pretrain(
            data_path,
            size="small",
            epochs=1,
            learning_rate=1e-3,
        )
        artifact = model.save(artifact_path, overwrite=True)

        print(f"Saved ArcLM artifact: {artifact.path}")
        print(f"Architecture: {model.inspect()['architecture_id']}")
        print(f"Vocabulary size: {model.config.vocab_size}")


if __name__ == "__main__":
    main()
