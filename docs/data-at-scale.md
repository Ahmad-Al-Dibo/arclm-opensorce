# Data at Scale

ArcLM `0.9.0` includes repeatable streaming dataset sources for JSONL, JSON, CSV, text files, directories, and user iterables.

```python
from arclm.data import open_dataset, analyze_dataset, split_dataset

with open_dataset("data/train.jsonl", format="jsonl", streaming=True, malformed="report") as dataset:
    report = analyze_dataset(dataset, schema="text")

splits = split_dataset(
    open_dataset("data/train.jsonl", format="jsonl", streaming=True),
    train=0.8,
    validation=0.1,
    test=0.1,
    strategy="hash",
    key="id",
    seed=42,
)
```

Malformed JSONL can either raise immediately or yield explicit `_error` records. No malformed rows are silently dropped.

Implemented scale APIs:

- `arclm.data.open_dataset`
- `arclm.data.DatasetSource`
- `arclm.data.shard_dataset`
- `arclm.data.split_dataset`
- `arclm.data.find_duplicates`
- `arclm.data.check_leakage`
- `arclm.data.analyze_dataset`
