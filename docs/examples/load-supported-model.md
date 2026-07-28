# Load A Supported Model

Native ArcLM checkpoints are officially supported.

```python
from arclm import load_model

loaded = load_model("model.pth", device="cpu")
print(loaded.config)
```

Inspect external models before loading:

```python
from arclm import inspect_model_source

print(inspect_model_source("gpt2").format_report())
```

