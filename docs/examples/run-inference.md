# Run Inference

```python
from arclm import load_model

loaded = load_model("model.pth", device="cpu")
text = loaded.predict("ArcLM", max_new_tokens=12, top_k=5)
print(text)
```

For a complete local script, see `examples/11_inference.py`.

