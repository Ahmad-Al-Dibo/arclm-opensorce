# Typed Configuration

Schema version `1` is represented by `arclm.config.ArcLMConfig`.

```python
from arclm.config import ArcLMConfig, DataConfig, ModelConfig, load_arclm_config

config = ArcLMConfig(
    data=DataConfig(path="data/train.jsonl", schema="text"),
    model=ModelConfig(name="gpt2"),
)

loaded = load_arclm_config("arclm.json")
```

Defaults:

- unknown fields fail unless `permissive=True`
- relative paths resolve relative to the config file
- environment variables expand only with `allow_env=True`
- secret-like values are redacted in `to_dict()` by default
- `model.trust_remote_code` defaults to `False`

CLI:

```bash
arclm config validate arclm.json
arclm config show arclm.json
arclm config migrate old.json --output arclm.json
```
