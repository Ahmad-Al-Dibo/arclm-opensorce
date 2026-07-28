# Dataset Splitting

Use `ProcessedDataset.split`.

```python
splits = dataset.split(train=0.8, validation=0.1, test=0.1, seed=42)
train_rows = splits["train"]
```

Native training data can also be split by `prepare_data(config)` using `Config.validation_split`.

