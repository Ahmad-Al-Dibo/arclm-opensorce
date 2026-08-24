# Run Inference

```python
from arclm import Model, Runtime

runtime = Runtime.auto(prefer="cpu")
loaded = Model.load("model.arcmodel", runtime=runtime)
text = loaded.generate("ArcLM", max_new_tokens=12)
print(text)
```

For a complete local script, see `examples/company_level/11_inference.py`.
