# Workflow Runner

ArcLM can run a local configuration-driven workflow:

```bash
arclm run arclm.json --dry-run
```

Python usage:

```python
from arclm.workflow import run_workflow

result = run_workflow("arclm.json", dry_run=True)
print(result.status)
```

Dry-run validates dataset access, schema compatibility, model support metadata, and output/run setup. It avoids training and avoids loading full model weights.

Minimal JSON configuration:

```json
{
  "run": {"name": "example", "output_dir": "runs"},
  "data": {"path": "data/train.jsonl", "format": "jsonl", "schema": "text"},
  "model": {"source": "hf-internal-testing/tiny-random-gpt2", "device": "cpu"},
  "training": {"enabled": false}
}
```

