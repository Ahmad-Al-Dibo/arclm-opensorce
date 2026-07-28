# Loading Models

## Native ArcLM

```python
from arclm import load_model

loaded = load_model("model.pth", device="cpu")
print(loaded.predict("ArcLM", max_new_tokens=16))
```

## Any Supported Source

```python
from arclm.models import inspect_model_support, load_model

support = inspect_model_support("gpt2", trust_remote_code=False)
print(support.summary())

model = load_model("gpt2", device="cpu", trust_remote_code=False)
print(model.predict("Hello", max_new_tokens=16))
```

The new facade does not enable `trust_remote_code` by default. Generic Hugging Face loading is compatible but unverified unless the model family is officially verified by ArcLM.
