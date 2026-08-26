"""Fine-tuning: attach native LoRA parameters and save an ArcLM adapter."""

from __future__ import annotations

import tempfile
from pathlib import Path

from arclm import Dataset, Lab, Trainer


def run(output_dir: str | Path | None = None) -> dict:
    root = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="arclm-lora-"))
    dataset = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"}])
    model = Lab().model(data=dataset, size="tiny")
    history = Trainer(
        model=model,
        dataset=dataset,
        method="lora",
        rank=2,
        alpha=4.0,
        target_modules=("head",),
        epochs=1,
        steps=1,
        batch_size=2,
        block_size=4,
        save_adapter=True,
        adapter_path=root / "adapter.arcadapter",
        shuffle=False,
    ).train()
    return {
        "steps": history["global_steps"],
        "adapter": history["adapter"],
        "trainable": history["metrics"]["trainable"]["trainable_parameters"],
    }


if __name__ == "__main__":
    print(run())
