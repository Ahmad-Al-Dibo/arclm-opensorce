# Cleaning Data

`ProcessedDataset.clean()` normalizes whitespace in string fields.

```python
dataset = DataProcessor.load("data.jsonl").clean()
```

For report-driven JSONL cleaning, use `PreprocessPipeline`.

