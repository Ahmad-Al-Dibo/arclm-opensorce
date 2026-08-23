# ArcLM vNext API Migration Table

Status: PARTIAL. "Removal" is the earliest intended major line, not a promise
that removal is ready now.

| Old API | New API | Status | Removal |
| --- | --- | --- | --- |
| `train_model(mode="pretrain", data=..., output=...)` | `Dataset.load(...)`, `Model.create(...)`, `Trainer(model, dataset).train()`, `model.save(...)` | MIGRATE | 1.x bridge, remove no earlier than 2.0 |
| `train_model(mode="finetune", checkpoint=...)` | `Model.load(...).finetune(...)` | PLANNED | 2.0 |
| `continue_native_training(...)` | `Trainer.resume(...).train()` | PLANNED | 2.0 |
| `train_native_model(...)` | `Trainer(model, dataset).train()` | MIGRATE | 2.0 |
| `fine_tune_native_model(...)` | `model.finetune(...)` | PLANNED | 2.0 |
| `train_sft(...)` | `model.finetune(data=..., method="sft"|"lora")` | PLANNED | 2.0 |
| `load_model("model.pth")` | `Model.load("model.arcmodel")` | MIGRATE | 2.0 after checkpoint converter exists |
| `load_any_model(...)` | `Model.load(...)` | MIGRATE | 2.0 |
| `predict(...)` | `Model.load(...).generate(...)` | MIGRATE | 2.0 |
| `Generator(...)` | `model.generate(...)` | INTERNALIZE | 2.0 |
| `Config(device="cuda")` | `Runtime.auto(prefer="cuda")` plus model/trainer options | MIGRATE | 2.0 |
| `get_device()` | `Runtime.auto()` | MIGRATE | 2.0 |
| `DeviceConfig.resolve()` | `Runtime.auto(...)` | MIGRATE | 2.0 |
| `prepare_data(config)` | `Dataset.load(...).inspect()` then `dataset.prepare(...)` | MIGRATE | 2.0 |
| `TextDataset` / `create_dataloader` | `Dataset.prepare(...)` | INTERNALIZE | 2.0 |
| `TokenizerFactory.create(...)` | `Dataset.prepare(tokenizer=...)` or future tokenizer spec | KEEP LOW-LEVEL | N/A |
| `Trainer(model, optimizer, criterion, config)` | `Trainer(model=Model, dataset=Dataset, ...)` | MIGRATE | 2.0 |
| `Trainer.save(config, ...)` | `model.save(..., layout=...)` and future `trainer.checkpoint(...)` | MIGRATE | 2.0 |
| `torch.save` checkpoint dictionaries | `.arcmodel` / `.arcckpt` / `.arcadapter` artifact APIs | REPLACE | 2.0 |
| `save_loaded_model(...)` | `model.save(...)` or `model.export(...)` | MIGRATE | 2.0 |
| `ModelSaveConfig` | `model.save(..., layout=..., include=...)` | MIGRATE | 2.0 |
| `inspect_model_source(...)` | `Model.inspect_source(...)` or `Model.load(..., inspect_only=True)` | PLANNED | 2.0 |
| `SmartLoader` | `Model.load()` resolver pipeline | MIGRATE | 2.0 |
| `arclm.registry.Registry` plugin registry | Future `arclm.registry` package for model/backend/artifact registries | MIGRATE | 2.0 |
| `arclm/api.py` helper module | Top-level `arclm` exports and future `arclm.api` package | MIGRATE | 2.0 |
| `MiniGPT` | `ArcLM` or `Model.create(...)` | DEPRECATED | Existing documented removal target |
| `checkpoint_is_compatible_for_tuining` | `checkpoint_is_compatible_for_tuning` | DEPRECATED | Existing documented removal target |

## Current Conflicts

- `arclm/api.py` blocks the target `arclm/api/` package.
- `arclm/registry.py` blocks the target `arclm/registry/` package.
- `arclm/data.py`, `arclm/dataset.py`, and `arclm/model.py` block simple package
  promotion for `arclm.data/` and `arclm.model/`.

Current bridge: expose `Lab`, `Dataset`, `Model`, `Trainer`, and `Runtime` from
the root package while implementation lives under `arclm.vnext` and conflict
modules remain compatibility shims.
