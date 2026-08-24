# Evaluate A Model

```python
from arclm import Dataset, Model, Runtime, calculate_metrics

loaded = Model.load("model.arcmodel", runtime=Runtime.auto(prefer="cpu"))
config = loaded.config
dataset = Dataset.load([{"text": "ArcLM evaluation text " * 12}])
prepared = dataset.prepare(tokenizer=loaded.tokenizer, block_size=config.block_size, batch_size=2)
metrics = calculate_metrics(loaded.model, prepared.train_loader, config, loaded.runtime.torch_device())
print(metrics.to_dict())
```

Use a real validation dataset for meaningful results.
