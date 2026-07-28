# Records And Schemas

Records are dictionaries that can be validated against formal schemas:

```python
from arclm.data import validate_records

report = validate_records(records, schema="conversation", strict=True)
print(report.is_valid)
```

Supported schemas are `text`, `prompt_completion`, `instruction`, and `conversation`.
