# Saving Processed Data

`ProcessedDataset` does not yet provide a save method. Save rows with standard Python:

```python
import json

with open("processed.jsonl", "w", encoding="utf-8") as f:
    for row in dataset.samples:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

`PreprocessPipeline.run` writes JSONL output directly.

