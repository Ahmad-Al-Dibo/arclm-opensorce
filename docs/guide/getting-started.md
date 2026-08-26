# Getting Started

## Installation

From the repository root:

```bash
pip install -e .
```

For development, tests, and docs:

```bash
pip install -e ".[dev]"
```

ArcLM currently supports Python `>=3.9,<3.13`. PyTorch and `safetensors` are
runtime dependencies. SentencePiece is optional and only needed for
`Tokenizer(strategy="sentence")`.

## A Tiny First Run

This example is derived from `examples/01_student_lab.py`.

```python
from arclm import Lab

lab = Lab()
dataset = lab.dataset(text="alpha beta gamma delta alpha beta gamma delta")
model = lab.model(data=dataset, size="tiny")
history = lab.train(epochs=1, steps=1, shuffle=False)
```

`history` is a dictionary with structured training information including
`global_steps`, `train_losses`, `tokens_processed`, `samples_processed`, and
stopping status.

## Inspecting Data

```python
report = dataset.inspect()
print(report.records, report.characters, report.estimated_tokens)
```

Datasets currently load text files, JSON, JSONL, or in-memory records.

## Next Steps

- Use [Student / Lab API](student-lab.md) for the simplest workflow.
- Use [Professional API](professional.md) for explicit production-like runs.
- Use [Research API](research.md) when replacing internal components.
