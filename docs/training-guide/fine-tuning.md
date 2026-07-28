# Fine-Tuning

Native fine-tuning uses an existing checkpoint:

```python
from arclm import train_model

result = train_model(
    mode="finetune",
    data="new-data.txt",
    output="finetuned.pth",
    checkpoint="base.pth",
    num_epochs=1,
)
```

Tokenizer compatibility is checked for native ArcLM checkpoints when required.

