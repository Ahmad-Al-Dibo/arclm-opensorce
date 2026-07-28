# Reproducible Pipelines

Use explicit seeds and save configuration:

```python
from arclm import Config
from arclm.config_loader import save_config_json

config = Config(seed=42, validation_split=0.1)
save_config_json(config, "config.json")
```

Prefer checking transformed rows before training and preserving tokenizer metadata in checkpoints.

