# Load A Supported Model

Native ArcLM `.arcmodel` artifacts are supported through the new public Model API.

```python
from arclm import Model, Runtime

loaded = Model.load("model.arcmodel", runtime=Runtime.auto(prefer="cpu"))
print(loaded.inspect())
```

Inspect external models before loading:

```python
from arclm import inspect_model_source

print(inspect_model_source("gpt2").format_report())
```
