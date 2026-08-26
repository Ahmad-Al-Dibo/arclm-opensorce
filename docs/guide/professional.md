# Professional API

The professional API is for repeatable training pipelines. It keeps the same
engine but makes configuration explicit.

This example is derived from `examples/02_professional_training.py`.

```python
import torch

from arclm import Dataset, Model, Runtime, Tokenizer, Trainer
from arclm.training import FineTuningConfig, TrainingConfig

dataset = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
validation = Dataset.load([{"text": "alpha beta gamma delta"}])
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
    save_artifact=True,
    artifact_path="professional.arcmodel",
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
```

## Configurable Surface

Professional workflows can configure:

- model architecture and model dimensions
- tokenizer strategy and vocabulary
- dataset and validation dataset
- runtime/device preference
- training config, steps, early stopping, checkpoints, and artifact output
- optimizer and scheduler injection
- callbacks and progress handling
- full fine-tuning or adapter/LoRA fine-tuning

`TrainingConfig.to_dict()` and `FineTuningConfig.to_dict()` return JSON-friendly
configuration reports.
