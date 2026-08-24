# ArcLM vNext Migration Notes

Status: PARTIAL.

No existing public API has been removed.

New transitional APIs:

```python
from arclm import Model, Runtime

runtime = Runtime.auto()
model = Model.create(architecture="arclm-native", vocab_size=100)
model.save("tiny.arcmodel")
reloaded = Model.load("tiny.arcmodel")
```

Existing APIs such as `train_model`, `Trainer`, `load_model`, `train_sft`, and
`load_any_model` remain unchanged. Future migration work should route these
facades into the vNext core incrementally and emit deprecation warnings only
after compatibility tests exist.

## Transitional Naming

`arclm.vnext` exists because `arclm/api.py` and `arclm/registry.py` currently
block the target `arclm.api` and `arclm.registry` package names.
