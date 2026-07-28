# Supported Models

See the full matrix in [Supported Models](../supported-models.md).

From Python:

```python
from arclm import get_supported_models

for capability in get_supported_models():
    print(capability.family, capability.status)
```

