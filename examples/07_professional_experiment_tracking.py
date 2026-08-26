"""Local experiment tracking and comparison."""

from __future__ import annotations

from pathlib import Path


def run(output_dir: str | Path = "outputs/experiment-tracking") -> dict:
    from arclm import Dataset, EvaluationEngine, Experiment, Lab, Runtime, Trainer

    root = Path(output_dir)
    dataset = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"}])
    validation = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
    model = Lab(runtime=Runtime.auto(prefer="cpu")).model(data=dataset, size="tiny")
    trainer = Trainer(
        model=model,
        dataset=dataset,
        validation_dataset=validation,
        epochs=1,
        steps=1,
        batch_size=1,
        block_size=4,
        shuffle=False,
    )
    history = trainer.train()
    evaluation = EvaluationEngine().evaluate(trainer=trainer)
    experiment = Experiment("tiny-baseline", root=root, seed=42).capture(
        model=model,
        dataset=dataset,
        trainer=trainer,
        history=history,
        evaluation=evaluation,
    )
    experiment.save()
    comparison = Experiment.compare(experiment, root=root)
    return {"experiment": experiment.to_dict(), "comparison": comparison.to_dict()}


if __name__ == "__main__":
    print(run())
