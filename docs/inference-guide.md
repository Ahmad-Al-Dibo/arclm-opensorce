# Inference Guide

Native:

```python
from arclm import load_model

loaded = load_model("model.pth", device="cpu")
print(loaded.predict("Once upon", max_new_tokens=20, temperature=0.8))
```

External causal LM:

```python
from arclm.models import load_model

bundle = load_model("gpt2", device="cpu", precision="float32")
print(bundle.predict("Once upon", max_new_tokens=20))
```

For chat-style prompts:

```python
messages = [{"role": "user", "content": "Explain ArcLM in one sentence."}]
print(bundle.predict(messages[-1]["content"], max_new_tokens=32))
```
