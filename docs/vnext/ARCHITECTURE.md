# ArcLM vNext Architecture

Status: UPDATED AFTER CORE REFACTOR SLICE.

The current vNext implementation starts with a narrow transitional core:

```text
arclm.vnext.Model
  -> arclm.vnext.ModelRegistry / ModelSpec
  -> arclm.runtime.Runtime
  -> arclm.artifacts.ArcModelArtifact
  -> existing arclm.model.ArcLM and arclm.tokenizer.Tokenizer
```

`arclm.vnext` is temporary. The target architecture wants `arclm.api` and
`arclm.registry` packages, but this repository already has `arclm/api.py` and
`arclm/registry.py` modules. Those modules need a compatibility migration before
package directories with the same names can exist.

## Pre-Refactor Repository Snapshot

This snapshot reflects the code inspected before the training/artifact/core
refactor.

| Area | Current implementation |
| --- | --- |
| Public Trainer | `arclm.vnext.trainer.Trainer` accepts `Trainer(model=Model, dataset=Dataset)` but calls `arclm.pipeline.build_trainer()` and therefore executes the legacy `arclm.trainer.Trainer` loop. The legacy constructor remains bridged through `LegacyTrainer`. |
| Old training loops | `arclm.trainer.Trainer`, `arclm.pipeline.train_model()`, `arclm.training.unified.UnifiedPipeline`, legacy examples, SFT helpers, diagnostics and checkpoint tests still use old training/checkpoint flows. |
| Torch serialization | Native `.arcmodel` currently writes `weights.pt` or `weights/shard-00001.pt` using `torch.save()` and reads with `torch.load()`. Legacy checkpoints and loaders also use `torch.save/load`. |
| Artifact sharding | `layout="sharded"` exists but writes one shard only and rejects multi-shard reads. |
| Model family logic | Registry only knows `arclm-native-causal-lm`. Family-specific external detection still lives in `arclm.models`, `arclm.external_inference`, `arclm.loaders.smart_loader`, examples and HF/SFT helpers. |
| Model construction | `arclm.vnext.Model.create/load` resolves `ModelSpec`, then directly constructs `arclm.model.ArcLM`. There is no enforced architecture-builder contract yet. |
| Runtime | Public `Runtime` hides `torch.device`, but model/training code still calls `runtime.torch_device()` directly. |
| Dataset | vNext `Dataset.prepare()` owns text/token semantics but still returns a Torch-backed dataloader from legacy `arclm.dataset.create_dataloader()`. |
| Lab | `Lab` delegates to `Dataset`, `Model`, `Trainer` and artifact APIs. It is intentionally not an independent engine. |
| Conceptual contracts | `Model`, `ModelSpec`, `ModelRegistry`, `Runtime`, `Dataset`, `Artifact`, backend, architecture, weight IO and adapters are documented concepts, but only `ModelSpec/Registry` and artifact manifest are partially enforced in code. |

## Current Implemented Shape

The vNext public API now routes the native path through ArcLM Core:

```text
arclm.Model / Dataset / Trainer / Runtime
  -> arclm.core contracts
  -> ModelRegistry / ModelSpec
  -> Architecture builder
  -> TrainingEngine + Strategy
  -> Artifact writer/reader + adapter writer/reader
  -> Torch backend for tensors/autograd/optimizers
```

| Area | Current implementation |
| --- | --- |
| Public Trainer | `Trainer(model=Model, dataset=Dataset).train(mode="pretrain")` uses `arclm.core.TrainingEngine`; it no longer calls `build_trainer()` or `LegacyTrainer`. The old positional constructor still bridges to legacy `Trainer` for compatibility only. |
| Training strategies | One engine executes lifecycle/forward/loss/backward/optimizer/scheduler/gradient hooks. `PretrainStrategy`, `FullFineTuneStrategy` and `AdapterStrategy` select mode-specific preparation and loss behavior. |
| Core contracts | `arclm.core.contracts` defines typed runtime-checkable contracts for model, spec, registry, runtime, dataset, backend, training engine, artifact, weight IO, mapper and adapter boundaries. |
| Backend boundary | `TorchBackend` owns optimizer creation, cross-entropy loss, backward, optimizer stepping, scheduler stepping, gradient clipping and scalar extraction. Core still uses Torch as the active backend. |
| ModelSpec | The native spec now carries architecture id, task, capabilities, configuration requirements, tokenizer requirements, runtime requirements, default adapter targets, architecture builder and weight mapper. |
| Artifact | `.arcmodel` uses ArcLM manifest/config/tokenizer/index metadata with safetensors tensor storage. Single, directory and true multi-shard layouts are implemented. |
| Adapter | `.arcadapter` stores native LoRA metadata and safetensors adapter tensors, and rejects incompatible base-model fingerprints. |
| Pretrained family | The implemented registry-resolved family is `arclm-native-causal-lm`: portable ArcLM-native pretrained/base artifacts. External GPT-2/Qwen/Llama loaders remain outside this Core path. |

## Current Principles Implemented

- ArcLM owns the public model lifecycle through `Model.create()`,
  `Model.load()`, `Model.save()`, `Model.generate()` and `Model.inspect()`.
- Runtime selection is represented as an ArcLM `Runtime`, not as a public
  `torch.device`.
- Model support is described by `ModelSpec` and capability statuses.
- `.arcmodel` is an ArcLM-owned artifact with a versioned manifest, config,
  tokenizer payload, tensor metadata, hashes and safetensors tensor storage.
- Torch remains the internal tensor/backend implementation for the native model.

## Current Limitations

- The vNext model wrapper supports only the native ArcLM causal LM.
- Legacy checkpoint APIs still use `torch.save/load`.
- External pretrained model resolution is still handled by legacy loaders, not
  by this new Core pipeline.
- No lazy/partial tensor loading yet.
