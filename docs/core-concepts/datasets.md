# Datasets

`DataProcessor.load(path)` returns a `ProcessedDataset`.

Supported input formats:

- `.json`
- `.jsonl`
- `.csv`
- `.txt`
- custom loader callable

The dataset is in memory. For large datasets, use the JSONL preprocessing pipeline or add streaming support before production use.

