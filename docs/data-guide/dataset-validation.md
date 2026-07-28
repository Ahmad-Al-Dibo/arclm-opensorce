# Dataset Validation

ArcLM validates `text`, `prompt_completion`, `instruction`, and `conversation` records.

```python
from arclm.data import validate_records

records = [{"text": "A valid training sample."}]
report = validate_records(records, schema="text", strict=True)
report.raise_for_errors()
print(report.summary())
```

Validation reports collect errors and warnings; they do not remove invalid records.
