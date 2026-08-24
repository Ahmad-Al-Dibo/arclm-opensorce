# Framework API

ArcLM `1.0.0` is organized around an ArcLM-owned route. The classes below are
the high-level framework surface for inspecting data, creating native models,
planning training, running training, saving artifacts, loading artifacts, and
generating text.

Use this page when you want the self-driving route. Use the lower-level API
pages when you need direct access to data processors, tokenizers, checkpoint
helpers, or Hugging Face compatibility paths.

## Main Flow

```python
from arclm import Dataset, Lab, Model, Runtime

runtime = Runtime.auto(prefer="cpu")
lab = Lab(runtime=runtime)

dataset = Dataset.load("data/train.jsonl")
report = lab.inspect(dataset)

model = lab.create(size="tiny", data=dataset)
plan = lab.plan(model, dataset, epochs=1)
history = lab.train(model, dataset)

model.save("model.arcmodel", overwrite=True)
loaded = Model.load("model.arcmodel", runtime=runtime)
print(loaded.generate("ArcLM", max_new_tokens=8))
```

## `Lab`

Import path: `from arclm import Lab`

`Lab` is the highest-level ArcLM interface. It keeps the latest runtime,
dataset, model, trainer, and decision history so a workflow can be inspected as
it moves from data to model to training.

| Method | Purpose |
| --- | --- |
| `Lab.inspect(target=None)` | Inspect a dataset target or return the current Lab state, including runtime, decisions, latest dataset, latest model, and latest trainer. |
| `Lab.create(task="causal-lm", size="small", data=None, **overrides)` | Create a native ArcLM causal model with inspectable defaults. Sizes include `tiny` and `small`; overrides can include model and tokenizer settings. |
| `Lab.plan(model, data, **options)` | Create a `TrainingPlan` without executing training. |
| `Lab.train(model, data, debug=False, **options)` | Train a model through the high-level `Trainer` and store the resulting history. |
| `Lab.pretrain(data, size="small", debug=False, **options)` | Load or accept a dataset, create a model, train it, and return the trained model. |

Typical `Lab.pretrain` options:

- Model options: `embed_dim`, `block_size`, `num_blocks`, `batch_size`,
  `vocab_size`, `tokenizer`.
- Training options: `epochs`, `batch_size`, `learning_rate`, `block_size`.

## `Dataset`

Import path: `from arclm import Dataset`

`Dataset` is the ArcLM-owned handle for high-level training data. It can load
local `txt`, `jsonl`, and `json` sources or accept in-memory records.

| Method | Purpose |
| --- | --- |
| `Dataset.load(source, format=None)` | Load text, JSONL, JSON, or iterable records into a dataset handle. |
| `Dataset.inspect()` | Return a lightweight report with source, format, bytes, record count, character count, estimated tokens, and warnings. |
| `Dataset.prepare(tokenizer=None, max_vocab=50000, block_size=8, batch_size=2, shuffle=True)` | Build or reuse a tokenizer, encode text, and create a next-token training dataloader. |
| `Dataset.text()` | Convert records into training text. |

Related records:

- `DatasetInspection`: report returned by `Dataset.inspect()`.
- `PreparedDataset`: prepared tokenizer, encoded tokens, dataloader, block size,
  and batch size.

## `Model`

Import path: `from arclm import Model`

`Model` is the ArcLM-owned native model wrapper. It carries the backend model,
ArcLM config, model registry spec, runtime, tokenizer, artifact, and base model
identity.

| Method | Purpose |
| --- | --- |
| `Model.create(architecture="arclm-native", tokenizer=None, runtime=None, **config_values)` | Create a native ArcLM model through registered architecture semantics. Requires `vocab_size` or a built tokenizer. |
| `Model.load(source, runtime=None)` | Load a native `.arcmodel` artifact or trusted legacy ArcLM checkpoint. |
| `Model.save(path, layout="auto", shard_size=None, overwrite=False)` | Save the model as an ArcLM-native `.arcmodel` artifact. |
| `Model.generate(prompt, max_new_tokens=20, temperature=0.0)` | Generate text through the attached tokenizer and native model. |
| `Model.inspect()` | Return architecture, capabilities, config, runtime, parameter count, artifact path, tokenizer type, and fingerprint. |
| `Model.get_tensor(name)` | Return a tensor by state-dict name. |
| `Model.set_tensor(name, value)` | Replace a tensor after shape validation. |
| `Model.attach_lora(rank=4, alpha=8.0, target_modules=None)` | Attach native LoRA adapters to selected target modules. |
| `Model.save_adapter(path, overwrite=False)` | Save attached native LoRA tensors as `.arcadapter`. |
| `Model.load_adapter(path)` | Load and attach a native `.arcadapter`. |
| `Model.finetune(data, method="lora", **kwargs)` | Fine-tune with the public trainer. Current high-level method is `lora`. |

## `Trainer` And `TrainingPlan`

Import paths:

```python
from arclm import Trainer, TrainingPlan
```

`Trainer` is the public high-level trainer bridge. When constructed with
`model=Model(...)` and `dataset=Dataset(...)`, it uses the ArcLM training
engine. When constructed with legacy positional arguments, it forwards to the
older trainer for compatibility.

| API | Purpose |
| --- | --- |
| `Trainer(model=model, dataset=dataset, epochs=1, batch_size=2, learning_rate=1e-3, block_size=8, ...)` | Create a high-level trainer and training plan. |
| `Trainer.make_plan()` | Build an inspectable `TrainingPlan`. |
| `Trainer.inspect()` | Return the plan and training history, or legacy trainer history. |
| `Trainer.train(mode="pretrain", debug=False, **kwargs)` | Train using the ArcLM training engine. Modes are `pretrain`, `full_finetune`, `adapter`, and `lora`. |
| `TrainingPlan.to_dict()` | Return a JSON-safe plan. |
| `TrainingPlan.summary()` | Print a Rich table when Rich is installed; otherwise print a plain install hint. |

## `Runtime` And `DeviceInfo`

Import path: `from arclm import Runtime`

`Runtime` is ArcLM's resolved execution plan. It chooses the backend, device,
precision, available devices, and warnings before model work starts.

| API | Purpose |
| --- | --- |
| `Runtime.auto(prefer="auto", precision="auto")` | Inspect the machine and choose CPU or CUDA. |
| `Runtime.device_name` | Return backend device string such as `cpu` or `cuda:0`. |
| `Runtime.torch_device()` | Return a `torch.device` for backend interoperability. |
| `Runtime.to_dict()` | Return a JSON-safe runtime report. |
| `DeviceInfo.to_dict()` | Return type, name, index, memory, and capabilities. |

Precision aliases:

- `auto`
- `float32`, `fp32`, `float`
- `float16`, `fp16`, `half`
- `bfloat16`, `bf16`

CPU requests for `float16` or `bfloat16` resolve to `float32`.

## `ModelRegistry`, `ModelSpec`, And Capabilities

Import paths:

```python
from arclm import ModelRegistry, ModelSpec
```

`ModelRegistry` is the ArcLM-owned registry for model architecture support. It
maps architecture IDs and aliases to `ModelSpec` records.

| API | Purpose |
| --- | --- |
| `ModelRegistry.register(spec)` | Register or replace a model spec. |
| `ModelRegistry.get(architecture_id)` | Return a spec by exact architecture ID. |
| `ModelRegistry.resolve(config_or_name)` | Resolve aliases such as `arclm`, `native`, or `arclm-native`. |
| `ModelRegistry.supported()` | Return all registered specs as support reports. |
| `ModelRegistry.supports(model, capabilities=None)` | Return support levels for all or selected capabilities. |
| `ModelSpec.supports(capability)` | Return the support level for one capability. |
| `ModelSpec.to_dict()` | Return a JSON-safe model spec report. |

The default registered architecture is `arclm-native-causal-lm`, with supported
native inference, pretraining, LoRA, saving, loading, and CPU execution.

Support labels include:

- `SUPPORTED`
- `PARTIAL`
- `EXPERIMENTAL`
- `PLANNED`
- `UNTESTED`
- `UNSUPPORTED`

## Lower-Level Companions

The high-level framework APIs work with the rest of ArcLM:

- `DataProcessor`, `ProcessedDataset`, and `DataPipeline` for structured data
  preparation.
- `Tokenizer` and `SentencePieceTokenizer` for text-to-token conversion.
- `ArcModelArtifact` and checkpoint helpers for native artifacts and safety.
- `inspect_model_source`, `load_any_model`, `train_sft`, and
  `arclm.models.load_model` for Hugging Face causal-LM paths through ArcLM
  wrappers.
- `evaluate`, `generate`, `run_workflow`, run metadata, diagnostics, and
  reproducibility helpers for the full route.
