# Precision

External Hugging Face loading accepts:

- `dtype="auto"`
- `dtype="float32"` or `dtype="fp32"`
- `dtype="float16"` or `dtype="fp16"`
- `dtype="bfloat16"` or `dtype="bf16"`

Example:

```python
from arclm.models import load_model

bundle = load_model("gpt2", precision="float32", device="cpu")
```

Native ArcLM does not expose a high-level mixed-precision training API in 0.8.0.dev0.
