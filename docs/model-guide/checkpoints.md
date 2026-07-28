# Checkpoints

Native checkpoints saved by `train_model` or `Trainer.save` include model weights, config, vocabulary mappings, and tokenizer metadata when provided.

```python
from arclm import train_model

result = train_model(mode="pretrain", data="data.txt", output="model.pth")
```

Only load trusted PyTorch checkpoints. PyTorch deserialization can execute code for unsafe files.

