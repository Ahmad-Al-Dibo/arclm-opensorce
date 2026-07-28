# Checkpoints

Native checkpoints saved by `train_model` or `Trainer.save` include model weights, config, vocabulary mappings, and tokenizer metadata when provided.

```python
from arclm import train_model

result = train_model(mode="pretrain", data="data.txt", output="model.pth")
```

Only load trusted PyTorch checkpoints. PyTorch deserialization can execute code for unsafe files.

ArcLM `0.9.0` adds a safer directory checkpoint inspection path:

```python
from arclm.checkpoints import inspect_checkpoint, verify_checkpoint

report = inspect_checkpoint("runs/example/checkpoint")
verify_checkpoint("runs/example/checkpoint")
```

Safe mode rejects legacy pickle-based `.pt`, `.pth`, `.ckpt`, and `.bin` files
unless a trusted loading policy is explicitly selected. See
[Checkpoint Specification](../checkpoint-specification.md).
