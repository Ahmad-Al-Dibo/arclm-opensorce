"""Student/Lab API: inspect data, create a model, and train."""

from __future__ import annotations

from arclm import Lab


def run() -> dict:
    lab = Lab()
    dataset = lab.dataset(text="alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta")
    model = lab.model(data=dataset, size="tiny")
    history = lab.train(epochs=1, steps=1, shuffle=False)
    return {
        "records": dataset.inspect().records,
        "architecture": model.inspect()["architecture_id"],
        "steps": history["global_steps"],
    }


if __name__ == "__main__":
    print(run())
