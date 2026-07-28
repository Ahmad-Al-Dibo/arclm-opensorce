# Evaluation Guide

ArcLM evaluation currently focuses on native checkpoints.

```python
from arclm import Config, calculate_metrics, create_dataloader, load_model

loaded = load_model("model.pth", device="cpu")
config = loaded.config
loader = create_dataloader(list(range(config.vocab_size)) * 4, config.block_size, 2)
metrics = calculate_metrics(loaded.model, loader, config, loaded.device)
print(metrics.to_dict())
```

This synthetic example validates the API shape; meaningful evaluation requires a real validation split.

