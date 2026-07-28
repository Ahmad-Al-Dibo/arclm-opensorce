# Checkpoint Specification

ArcLM checkpoint format version `1` is a directory with a manifest:

```text
checkpoint/
├── manifest.json
├── hashes.json
├── model/
│   ├── config.json
│   └── model.safetensors
├── tokenizer/
└── training/
    ├── state.json
    ├── optimizer.pt
    └── scheduler.pt
```

Safe mode inspects manifests and verifies hashes without loading PyTorch pickle
payloads. Optimizer and scheduler files may still require pickle when restored,
so they are only allowed under an explicit trusted policy.

```python
from arclm.checkpoints import inspect_checkpoint, verify_checkpoint

report = inspect_checkpoint("runs/example/checkpoint")
verify_checkpoint("runs/example/checkpoint")
```

Trust modes:

| Mode | Pickle | Remote code | Default |
| --- | --- | --- | --- |
| `safe` | rejected | rejected | yes |
| `trusted_local` | allowed for trusted local files | rejected | no |
| `legacy_unsafe` | allowed | allowed | no |
