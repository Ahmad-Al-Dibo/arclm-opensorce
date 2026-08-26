# Artifacts

ArcLM model artifacts use the `.arcmodel` format. The format currently stores:

- manifest metadata
- ArcLM artifact/schema version information
- architecture ID, version, metadata, and required capabilities
- model config
- tokenizer payload
- safetensors weights
- integrity hashes
- training metadata when provided

This example is derived from `examples/04_save_load_arcmodel.py`.

```python
from arclm import Dataset, Lab, Model

dataset = Dataset.load([{"text": "one two three one two three"}])
model = Lab().model(data=dataset, size="tiny")
artifact = model.save("tiny.arcmodel", overwrite=True)
loaded = Model.load(artifact.path)
```

Native LoRA adapters can be saved as `.arcadapter` through `model.save_adapter`
or `Trainer(..., save_adapter=True, adapter_path=...)`.

## Planned Artifact Work

The architecture supports richer artifact metadata. Additional migration policy,
format evolution, and broader sharding workflows are planned.
