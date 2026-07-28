# Inference

Native inference:

```python
from arclm import load_model

loaded = load_model("model.pth", device="cpu")
print(loaded.predict("ArcLM", max_new_tokens=16))
```

External causal-LM inference:

```python
from arclm.models import load_model

bundle = load_model("gpt2", device="cpu")
print(bundle.predict("Hello", max_new_tokens=16))
```

External inference requires model downloads and optional dependency compatibility.
