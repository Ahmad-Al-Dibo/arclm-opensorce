# Resuming Training

Use `mode="continue_training"` with a compatible native checkpoint:

```python
from arclm import train_model

result = train_model(
    mode="continue_training",
    data="data.txt",
    output="continued.pth",
    checkpoint="previous.pth",
)
```

Resume requires restorable tokenizer metadata or an explicit tokenizer.

