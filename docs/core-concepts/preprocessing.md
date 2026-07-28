# Preprocessing

`PreprocessPipeline` reads JSONL rows, cleans text, filters invalid samples, deduplicates, and writes reports.

```python
from arclm.preprocess import PreprocessConfig, PreprocessPipeline

config = PreprocessConfig(min_chars=20, allowed_languages=["en"])
report = PreprocessPipeline(config).run("raw.jsonl", "clean.jsonl", "reports/preprocess")
```

This pipeline currently supports JSONL input through `read_jsonl`.

