# Training and Fine-Tuning

ArcLM owns its training lifecycle. The current engine handles:

- epochs and deterministic per-epoch steps
- optional global `max_steps` safety cap
- batching and forward-pass coordination
- loss and backward pass
- gradient accumulation
- optimizer and scheduler coordination
- validation
- checkpoint/resume
- metrics and callbacks
- early stopping
- artifact export

## Deterministic Steps

```python
from arclm.training import TrainingConfig

config = TrainingConfig(epochs=3, steps_per_epoch=15)
```

This runs up to `45` training steps. `max_steps` is a global cap, not a
per-epoch value.

## Early Stopping

```python
config = TrainingConfig(
    epochs=10,
    steps_per_epoch=5,
    early_stopping=True,
    early_stopping_patience=1,
    early_stopping_metric="validation_loss",
)
```

If no validation dataset is provided, `validation_loss` falls back to the epoch
training loss for early stopping.

## Fine-Tuning

Full fine-tuning:

```python
Trainer(model=model, dataset=dataset, method="full_finetune").train()
```

LoRA-style adapter fine-tuning:

```python
Trainer(
    model=model,
    dataset=dataset,
    method="lora",
    rank=2,
    alpha=4.0,
    target_modules=("head",),
    steps=1,
).train()
```

ArcLM does not require Transformers or PEFT for this native adapter path.
