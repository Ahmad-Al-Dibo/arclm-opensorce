"""Professional API: explicit tokenizer, runtime, model, validation, and artifact output."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from arclm import Dataset, Model, Runtime, Tokenizer, Trainer
from arclm.training import FineTuningConfig, TrainingConfig


TEXT = "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"


def run(output_dir: str | Path | None = None) -> dict:
    root = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="arclm-professional-"))
    dataset = Dataset.load([{"text": TEXT}])
    validation = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
    tokenizer = Tokenizer(strategy="word", max_vocab=16).build(dataset.text())
    runtime = Runtime.auto(prefer="cpu")
    model = Model.create(
        architecture="arclm-native",
        tokenizer=tokenizer,
        runtime=runtime,
        embed_dim=8,
        block_size=4,
        num_blocks=1,
        batch_size=2,
    )
    config = TrainingConfig(
        epochs=1,
        batch_size=2,
        block_size=4,
        steps_per_epoch=1,
        learning_rate=5e-4,
        validation_interval=1,
        checkpoint_dir=root / "checkpoints",
        artifact_path=root / "professional.arcmodel",
        save_artifact=True,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.learning_rate)
    history = Trainer(
        model=model,
        dataset=dataset,
        validation_dataset=validation,
        config=config,
        fine_tuning=FineTuningConfig(method="pretrain"),
    ).train(optimizer=optimizer)
    return {
        "steps": history["global_steps"],
        "validation_losses": len(history["validation_losses"]),
        "artifact": history["artifact"],
    }


if __name__ == "__main__":
    print(run())
