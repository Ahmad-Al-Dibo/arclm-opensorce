# Training Configuration

Use `Config` or pass keyword overrides to `train_model`.

```python
from arclm import Config, train_model

config = Config(
    tokenizer_type="word",
    max_vocab=1000,
    embed_dim=64,
    block_size=32,
    num_blocks=2,
    batch_size=8,
    num_epochs=1,
    device="cpu",
)

result = train_model("pretrain", "data.txt", "model.pth", config=config)
```

Important fields include `embed_dim`, `block_size`, `num_blocks`, `dropout`, `batch_size`, `num_epochs`, `learning_rate`, `validation_split`, `tokenizer_type`, and `model_path`.

