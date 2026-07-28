# Security

ArcLM keeps remote code execution disabled by default in the new model facade and workflow runner. Checkpoint loading that uses PyTorch deserialization remains trusted-input only.

Utilities:

```python
from arclm.security import artifact_digest, scan_for_secrets

report = scan_for_secrets(["README.md"])
print(report.is_valid)
```

The scanner is a small guardrail for common accidental secrets. It does not prove that a repository is secure.

