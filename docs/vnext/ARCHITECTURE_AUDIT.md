# ArcLM vNext Architecture Audit

Status: PARTIAL Phase 0 audit, based on the open-source repository at
`ArcLM-opensorce/ArcLM-opensorce` and the local special reference repository at
`ArcLM-Special-version`.

## Repositories Inspected

| Repository | Role | Observed Shape |
| --- | --- | --- |
| Open source ArcLM | vNext implementation foundation | Flat `arclm/` package with legacy-compatible root exports, native tiny GPT-style model, training/data/tokenizer utilities, external HF/SFT helpers, CLI, docs and tests. |
| Special ArcLM | Capability mine/reference | Larger `src/arclm/` package with canonical domains for datasets, training, finetuning, inference, lifecycle artifacts, PEFT, security, web, search, distributed helpers, examples and standalone distributions. |

## Open-Source Package Structure

The open-source package currently has these primary areas:

| Area | Files/Packages | Notes |
| --- | --- | --- |
| Public root | `arclm/__init__.py` | Broad eager root import. Imports `torch` directly and re-exports most package concepts. |
| Native model | `arclm/model.py` | `ArcLM`, `MiniGPT`, `SelfAttention`, `GPTBlock`; single compact causal LM. |
| Training | `arclm/trainer.py`, `arclm/pipeline.py`, `arclm/training/*` | Multiple training entry points. `pipeline.train_model()` is the main high-level native path; `training/unified.py` is a separate older unification attempt. |
| Data | `arclm/data.py`, `data_processor.py`, `data_pipeline.py`, `data_quality.py`, `data_sources.py`, `schemas.py`, `preprocess/*` | Local text/token workflows plus JSON/JSONL/schema helpers and preprocessing utilities. |
| Tokenizers | `arclm/tokenizer.py`, `arclm/tokenizers/__init__.py`, `tokenization.py` | Word tokenizer and SentencePiece tokenizer. |
| Inference/loading | `arclm/inference.py`, `external_inference.py`, `loaders/*`, `models/__init__.py` | Native checkpoint loader plus HF/PEFT/safetensors/state-dict loading helpers. |
| Fine-tuning | `arclm/sft.py`, `instruction_dataset.py`, `trainer.freeze_layers()` | HF SFT path with optional PEFT LoRA; native SFT uses existing Trainer batch masks. |
| Runtime/device | `arclm/resources.py`, scattered `torch.cuda` calls | Device selection exists but is not yet one central ArcLM runtime. |
| Persistence | `Trainer.save()`, `inference.load_model()`, `external_inference.save_loaded_model()` | Native checkpoints are `torch.save()` dictionaries. External save delegates to HF `save_pretrained()`. |
| Diagnostics/support | `diagnostics.py`, `evaluation.py`, `supported_models.py`, `certification.py`, `doctor.py` | Useful reports but not yet unified under a vNext inspection layer. |
| Compatibility | Deprecated aliases and typo-preserving API such as `checkpoint_is_compatible_for_tuining` | Must be preserved until migration routes exist. |

## Special Repository Structure

The special version is much broader. Useful reference areas include:

| Area | Canonical Special Module | vNext Relevance |
| --- | --- | --- |
| Lazy public API | `src/arclm/__init__.py` | Good root import pattern: root exports are lazy and avoid pulling heavy stacks immediately. |
| Datasets | `src/arclm/datasets/*` | Cleaner canonical dataset package with validation, reports, streaming, cleaning and dedup. |
| Training | `src/arclm/training/trainer.py`, `config.py`, `state.py`, `precision.py` | More complete Trainer core with callbacks, precision, checkpoint resume and newer direct API. |
| Fine-tuning | `src/arclm/finetuning/*` | Canonical SFT formatting, label masks, collator, full finetuning flow. |
| PEFT | `src/arclm/peft/*` | Native LoRA, adapter artifact concepts and QLoRA boundary work. |
| Lifecycle artifacts | `src/arclm/lifecycle/*` | Existing artifact manifest ideas worth porting, not merging wholesale. |
| Runtime diagnostics | `src/arclm/diagnostics_runtime.py`, `distributed.py` | Runtime/capability inspection ideas. Distributed remains experimental. |
| Web/Studio | `src/arclm/web/*`, `web/src/*` | Product UI and local workflow proof, but out of scope for the first vNext core slice. |
| Security | `src/arclm/security/*` | Strong capability mine for later artifact integrity and secure loading. |
| Search/tracking | `src/arclm/search`, `tracking.py` | Useful later for experiments, not Phase 1 core. |

## Public APIs

Open-source root `arclm.__all__` currently exposes a very wide API: model
classes, config schemas, dataset records and helpers, checkpoint helpers,
diagnostics, inference, external model loading, supported-model metadata,
logging/cache/runs/workflow utilities, regularization, SFT, tokenizer classes
and training classes.

Important public entry points:

| Category | Current API |
| --- | --- |
| Native training | `train_model`, `build_model`, `build_trainer`, `Trainer`, `Config` |
| Native inference | `load_model`, `predict`, `LoadedModel`, `Generator` |
| External loading | `load_any_model`, `load_external_for_inference`, `inspect_model_source`, `save_loaded_model` |
| HF SFT | `train_sft`, `SFTTrainingResult` |
| Data | `prepare_data`, `DataProcessor`, `DataPipeline`, `validate_records`, `open_dataset` |
| Tokenizers | `Tokenizer`, `SentencePieceTokenizer`, `TokenizerFactory`, `create_tokenizer` |
| Devices | `get_device`, `DeviceConfig`, `resource_info`, `normalize_device` |

## Current Flows

### Native Pretraining

`train_model(mode="pretrain")` builds a `Config`, calls `prepare_data()`, builds
`ArcLM`, creates `Trainer`, runs the loop, then writes a native checkpoint through
`Trainer.save()`.

### Native Fine-Tuning / Continue Training

`train_model(mode="finetune" | "continue_training")` loads a checkpoint through
`load_external_model()`, optionally restores tokenizer metadata, validates
compatibility, adapts weights/config with `adapt_for_training()`, freezes layers
through `Trainer.freeze_layers()`, then uses the same `Trainer`.

### Hugging Face SFT

`train_sft()` imports Transformers lazily, loads `AutoTokenizer` and
`AutoModelForCausalLM`, optionally wraps with PEFT LoRA, builds an ArcLM-owned SFT
dataset/collator, runs a local Torch training loop, and saves with
`save_pretrained()`.

### Model Loading

Native `load_model()` expects an ArcLM/PyTorch checkpoint and reconstructs
`Config`, `ArcLM`, tokenizer metadata and `Generator`.

`external_inference.load_any_model()` inspects paths/model IDs, then delegates to
native loading, HF loading, PEFT adapter loading, or state-dict adaptation.

### Dataset/Tokenizer

`prepare_data()` reads whitespace tokens, splits train/validation, builds either
the word tokenizer or SentencePiece tokenizer, encodes tokens and returns loaders.
Data inspection and quality utilities exist separately and are not yet a required
stage in the training plan.

## External Dependency Exposure

Current public architecture exposes backend concepts in several places:

| Dependency | Public Exposure |
| --- | --- |
| `torch` | Root `get_device()` returns `torch.device`; `load_training_checkpoint()` calls `torch.load`; `Trainer` requires Torch optimizer/criterion; configs accept raw device strings; native checkpoints are `torch.save` dictionaries. |
| `transformers` | `train_sft`, `external_inference`, examples and docs expose HF `from_pretrained`, `device_map`, `save_pretrained` semantics. |
| `peft` | `train_sft(use_lora=True)` and adapter loading expose PEFT-oriented target module names and adapter folder behavior. |

These are acceptable implementation capabilities, but vNext should move their
semantics behind ArcLM concepts such as `Runtime`, `ModelSpec`, `Model.load()`,
`Artifact`, `Adapter`, and capability reports.

## Architectural Inconsistencies

- `arclm/api.py` and `arclm/registry.py` are files, so the target packages
  `arclm/api/` and `arclm/registry/` cannot be introduced until those legacy
  modules are migrated or renamed.
- Root imports are eager in open source, unlike special's lazy root exports.
- There are multiple training surfaces: `Trainer`, `pipeline.train_model`,
  `training.engine.train`, `training.unified.UnifiedPipeline`, HF SFT loop and
  external inference wrappers.
- Save/load semantics mix native checkpoints, resumable training state, HF export
  folders and adapter folders.
- Device/runtime selection is scattered across root helpers, config validation,
  resources, inference, SFT and external loading.
- Model-family detection sometimes uses string/name heuristics in source
  inspection. vNext should prefer config metadata.

## Dead or Suspicious Code

- Several files contain a UTF-8 BOM that caused naive AST parsing errors.
- Some log strings contain mojibake from old checkmark symbols.
- `checkpoint_is_compatible_for_tuining` is a misspelled public alias retained
  for compatibility.
- `training/unified.py` overlaps with `pipeline.py` and `trainer.py`.
- `pipeline_v2.py` exists as a compatibility/deprecated surface.

## Phase 0 Decision

The open-source repository remains the implementation base. The special
repository should be mined for:

- lazy root export design;
- canonical dataset and finetuning APIs;
- Trainer checkpoint/resume concepts;
- lifecycle artifact manifest concepts;
- native LoRA and adapter artifact behavior;
- runtime diagnostics and security boundaries.

It should not be merged wholesale.
