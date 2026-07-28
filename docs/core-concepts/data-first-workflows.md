# Data-First Workflows

ArcLM treats data preparation as the start of the framework workflow, not a side task. Use `DataProcessor` for small in-memory datasets and `PreprocessPipeline` for JSONL cleaning reports.

Recommended order:

```text
Load -> Clean -> Filter/Validate -> Transform -> Tokenize -> Train/Evaluate/Infer
```

