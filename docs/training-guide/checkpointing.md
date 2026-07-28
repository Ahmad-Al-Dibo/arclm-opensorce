# Checkpointing

`Config.checkpoint_interval` and `Config.checkpoint_batch_interval` control periodic native checkpoint saves.

```python
from arclm import Config

config = Config(checkpoint_batch_interval=100, model_path="checkpoints/model.pth")
```

Use checkpoints saved by ArcLM to preserve tokenizer metadata.

