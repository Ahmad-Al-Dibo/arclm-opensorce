# Evaluate A Model

```python
from arclm import calculate_metrics, create_dataloader, load_model

loaded = load_model("model.pth", device="cpu")
config = loaded.config
loader = create_dataloader(list(range(config.vocab_size)) * 4, config.block_size, 2)
metrics = calculate_metrics(loaded.model, loader, config, loaded.device)
print(metrics.to_dict())
```

Use a real validation dataset for meaningful results.

