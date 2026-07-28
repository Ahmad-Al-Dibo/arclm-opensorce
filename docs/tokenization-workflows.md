# Tokenization Workflows

Use `arclm.tokenization.tokenize_dataset` for schema-aware tokenization and deterministic JSON cache entries.

```python
from arclm.tokenization import tokenize_dataset

result = tokenize_dataset(
    [{"prompt": "Say hello", "completion": "Hello"}],
    tokenizer="gpt2",
    schema="prompt_completion",
    max_length=128,
    truncation=True,
    padding=False,
    prompt_masking=True,
    cache_dir=".arclm/cache",
)

print(result.cache_hit)
```

Supported schemas are `text`, `prompt_completion`, `instruction`, and `conversation`.

Cache keys include the dataset fingerprint, tokenizer identity, tokenization configuration, and ArcLM version. Cache files are JSON and are written atomically where practical.

