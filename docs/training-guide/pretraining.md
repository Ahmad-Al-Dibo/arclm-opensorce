# Pretraining

```python
from arclm import train_model

result = train_model(
    mode="pretrain",
    data="data.txt",
    output="model.pth",
    tokenizer_type="word",
    max_vocab=5000,
    num_epochs=1,
)
```

Pretraining builds a tokenizer from the training text unless an existing tokenizer is supplied.

