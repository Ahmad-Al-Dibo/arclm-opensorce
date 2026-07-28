# Filtering Data

Use `ProcessedDataset.filter(predicate)`.

```python
dataset = (
    DataProcessor.load("data.jsonl")
    .clean()
    .filter(lambda row: bool(row.get("text")) and len(row["text"]) > 20)
)
```

The predicate receives each record and returns `True` to keep it.

