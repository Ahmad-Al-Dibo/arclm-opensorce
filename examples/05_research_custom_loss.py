"""Research API: replace the loss through a custom strategy."""

from __future__ import annotations

from arclm import Dataset, Lab
from arclm.research import BaseStrategy, NextTokenLoss, Trainer, TrainingConfig


class HalfScaleLoss:
    """Small example loss wrapper for research experiments."""

    def __init__(self):
        self.calls = 0
        self.base = NextTokenLoss()

    def __call__(self, *, model, batch, engine):
        self.calls += 1
        return self.base(model=model, batch=batch, engine=engine) * 0.5


class HalfLossStrategy(BaseStrategy):
    """Strategy using the custom loss while preserving the ArcLM engine loop."""

    def __init__(self, loss: HalfScaleLoss):
        super().__init__(name="half_loss", loss_function=loss)


def run() -> dict:
    dataset = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta alpha beta gamma delta"}])
    model = Lab().model(data=dataset, size="tiny")
    loss = HalfScaleLoss()
    strategy = HalfLossStrategy(loss)
    history = Trainer(
        model=model,
        dataset=dataset,
        config=TrainingConfig(epochs=1, batch_size=2, block_size=4, steps_per_epoch=1, shuffle=False),
        strategy=strategy,
    ).train()
    return {
        "strategy": history["strategy"],
        "steps": history["global_steps"],
        "loss_calls": loss.calls,
    }


if __name__ == "__main__":
    print(run())
