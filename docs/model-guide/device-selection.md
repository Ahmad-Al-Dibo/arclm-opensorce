# Device Selection

Native ArcLM uses `Config.device` or the `device` parameter for loading:

```python
from arclm import Config, get_device

config = Config(device="cpu")
print(get_device())
```

External loading accepts `device`, `device_map`, `max_memory`, `load_in_8bit`, and `load_in_4bit` through `ExternalModelConfig` or `load_any_model`.

Hardware behavior depends on PyTorch, Transformers, and the selected model.

