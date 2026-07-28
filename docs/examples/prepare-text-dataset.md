# Prepare A Text Dataset

```python
from arclm import DataProcessor

dataset = (
    DataProcessor.load("data.txt")
    .clean()
    .filter(lambda row: len(row.get("text", "")) > 10)
    .transform(format="pretraining")
)

print(dataset.samples[0]["text"])
```

Works with TXT files where each non-empty line becomes one `{ "text": ... }` record.

