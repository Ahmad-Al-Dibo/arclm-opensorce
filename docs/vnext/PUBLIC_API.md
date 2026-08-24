# ArcLM vNext Public API

Status: PARTIAL migration design plus initial implementation.

ArcLM should have one recommended architecture with three levels of control.
The levels are public API layers over the same core model, dataset, runtime,
training and artifact objects.

## Level 1: Lab API

Audience: students, beginners, quick experiments.

Implemented:

```python
from arclm import Lab

lab = Lab()
report = lab.inspect("data.txt")
model = lab.pretrain("data.txt", size="tiny", epochs=1)
model.save("student-model.arcmodel")
lab.inspect()
```

Current `Lab` methods:

| API | Status | Notes |
| --- | --- | --- |
| `Lab()` | PARTIAL | Creates an inspectable runtime and decision log. |
| `lab.inspect(path)` | PARTIAL | Loads and inspects a dataset. |
| `lab.inspect()` | PARTIAL | Reports latest dataset/model/trainer decisions. |
| `lab.create(task="causal-lm", size="small")` | PARTIAL | Creates native ArcLM model defaults. |
| `lab.plan(model, data)` | PARTIAL | Returns an inspectable training plan. |
| `lab.train(model, data)` | PARTIAL | Delegates to the shared Trainer bridge. |
| `lab.pretrain(data)` | PARTIAL | Convenience route: load data, create model, train. |

## Level 2: High-Level API

Audience: developers and ML engineers who want power without tensor-level
management.

Implemented:

```python
from arclm import Dataset, Model, Trainer, Runtime

runtime = Runtime.auto()
data = Dataset.load("data.txt")
model = Model.create(architecture="arclm-native", tokenizer=data.prepare().tokenizer)
trainer = Trainer(model=model, dataset=data, epochs=1)
trainer.train()
model.save("./model", layout="directory")
```

Current high-level APIs:

| API | Status | Notes |
| --- | --- | --- |
| `Dataset.load(path)` | PARTIAL | Supports local TXT/JSON/JSONL. |
| `Dataset.inspect(path)` / `dataset.inspect()` | PARTIAL | Reports format, size, records, characters and estimated tokens. |
| `Dataset.prepare(...)` | PARTIAL | Word-tokenizer next-token preparation for native causal LM. |
| `Model.create(...)` | PARTIAL | Native ArcLM causal LM only. |
| `Model.load(path)` | PARTIAL | `.arcmodel` artifacts only. |
| `model.save(path, layout=...)` | PARTIAL | Supports `single`, `directory`, and one-shard `sharded`. |
| `model.generate(prompt)` | PARTIAL | Native deterministic/sampling generation. |
| `model.inspect()` | PARTIAL | Model, runtime, artifact and capability report. |
| `Trainer(model, dataset)` | PARTIAL | Public bridge over the existing tested Trainer loop. |
| `Runtime.auto()` | IMPLEMENTED | ArcLM-owned runtime object with Torch backend hidden behind it. |

## Level 3: Low-Level API

Audience: researchers and advanced users.

Implemented:

| API | Status | Notes |
| --- | --- | --- |
| `model.get_tensor(name)` | PARTIAL | Reads named state-dict tensor. |
| `model.set_tensor(name, value)` | PARTIAL | Replaces an existing tensor with shape validation. |
| `model.inspect()` | PARTIAL | First structured model report. |
| `artifact.inspect()` | PARTIAL | Manifest/layout/hash report without exposing raw tensors. |

Planned:

- layer/component inspection;
- component replacement/edit transactions;
- tensor mapping abstraction independent of raw backend state-dict names;
- weight mapper APIs for external model import.

## Internal-Only API

These should become implementation details or compatibility bridges:

| API | Direction |
| --- | --- |
| `arclm.vnext.*` | Transitional namespace; public imports should be from `arclm`. |
| `arclm.pipeline.train_model` internals | Route through `Trainer`/`Model`/`Dataset` later. |
| `build_model`, `build_trainer` | Keep as low-level/internal compatibility, not recommended beginner path. |
| `arclm.loaders.*` | Move behind `Model.load()` resolver pipeline. |
| raw `torch.save`/`torch.load` usage | Replace as public persistence contract with artifacts. |

## Deprecated API

No new runtime deprecation warnings were added in this phase, because the old
test suite and examples still rely on legacy routes. The migration table in
`API_MIGRATION.md` marks intended deprecations.
