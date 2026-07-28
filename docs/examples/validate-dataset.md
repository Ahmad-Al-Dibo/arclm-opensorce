# Validate A Dataset

```python
from arclm.data import validate_records

records = [{"text": "A valid row."}, {"text": ""}]
report = validate_records(records, schema="text", strict=True)
print(report.summary())
```
