"""Artifacts: save and load an ArcLM-native .arcmodel."""

from __future__ import annotations

import tempfile
from pathlib import Path

from arclm import Dataset, Lab, Model
from arclm.artifacts import ArcModelArtifact


def run(output_dir: str | Path | None = None) -> dict:
    root = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="arclm-artifact-"))
    dataset = Dataset.load([{"text": "one two three one two three one two three"}])
    model = Lab().model(data=dataset, size="tiny")
    artifact = model.save(root / "tiny.arcmodel", overwrite=True)
    loaded = Model.load(artifact.path)
    report = ArcModelArtifact(artifact.path).inspect()
    return {
        "path": str(artifact.path),
        "loaded_architecture": loaded.inspect()["architecture_id"],
        "tokenizer": report["tokenizer_metadata"]["strategy"],
    }


if __name__ == "__main__":
    print(run())
