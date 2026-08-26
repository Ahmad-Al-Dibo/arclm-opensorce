# Student / Lab API

`Lab` is the beginner-facing API. It hides tokenizer construction, model config,
runtime selection, optimizer setup, callbacks, artifact internals, and registry
details unless you deliberately move to a lower-level API.

```python
from arclm import Lab

lab = Lab()
dataset = lab.dataset(text="alpha beta gamma delta alpha beta gamma delta")
model = lab.model(data=dataset)
result = lab.train(epochs=1, steps=1, shuffle=False)
```

## What Lab Does

- `lab.dataset(...)` loads or creates an ArcLM `Dataset`.
- `lab.model(...)` creates a native tiny/small model and prepares a tokenizer
  from the dataset when available.
- `lab.train(...)` delegates to the same public `Trainer` used by professional
  workflows.
- `lab.fine_tune(...)` delegates to `Trainer(..., method=...)`.

The training engine underneath is not duplicated. `Lab` is a convenience layer
over `Dataset`, `Tokenizer`, `Model`, `Runtime`, and `Trainer`.
