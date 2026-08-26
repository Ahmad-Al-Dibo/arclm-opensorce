# Research API

`arclm.research` is a curated extension namespace. It exposes intentional
interfaces and components rather than random private internals.

This example is derived from `examples/05_research_custom_loss.py`.

```python
from arclm import Dataset, Lab
from arclm.research import BaseStrategy, NextTokenLoss, Trainer, TrainingConfig

class HalfScaleLoss:
    def __init__(self):
        self.base = NextTokenLoss()

    def __call__(self, *, model, batch, engine):
        return self.base(model=model, batch=batch, engine=engine) * 0.5

strategy = BaseStrategy(name="half_loss", loss_function=HalfScaleLoss())
dataset = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
model = Lab().model(data=dataset, size="tiny")
history = Trainer(
    model=model,
    dataset=dataset,
    config=TrainingConfig(epochs=1, batch_size=2, block_size=4, steps_per_epoch=1),
    strategy=strategy,
).train()
```

## Extension Points

The research namespace includes:

- architecture contracts and the architecture registry
- tokenizer engine contracts and `Tokenizer.register_engine(...)`
- training strategies, loss functions, evaluator, callbacks, and events
- checkpoint manager and training engine
- backend/runtime contracts

Researchers can replace a strategy, loss, tokenizer engine, evaluator, callback,
checkpoint manager, runtime, or architecture while still using the shared ArcLM
engine.
