# ArcLM vNext Feature Inventory

Status: PARTIAL Phase 0 inventory. Decisions describe the current vNext
direction, not final support guarantees.

| Capability | Open Source | Special | vNext Decision | Notes |
| --- | --- | --- | --- | --- |
| Native causal LM | Yes | Yes | KEEP | Compact `ArcLM` is the first native architecture for the vertical slice. |
| Pretraining | Yes | Yes | REDESIGN | Preserve behavior, but route through inspect/validate/resolve/plan/execute. |
| Native fine-tuning | Basic | Advanced | REDESIGN | Keep one Trainer engine with strategy/config differences. |
| HF SFT | Yes | Yes | WRAP | Keep optional Transformers capability behind ArcLM fine-tuning APIs. |
| LoRA | External PEFT path | Native and external references | PORT | First port should target native ArcLM Linear layers, then wrap PEFT for external models. |
| QLoRA | Minimal/optional | Experimental | INVESTIGATE | Keep as capability-specific experimental path. |
| Dataset preparation | Yes | Richer canonical package | PORT | Special `datasets` ideas should inform vNext data inspection and reports. |
| Dataset inspection | Partial | Rich | PORT | Needed before high-level `Lab.pretrain()`. |
| Tokenization | Word and SentencePiece | Word, SentencePiece, auto concepts | KEEP | Keep current tokenizers; introduce tokenizer specs later. |
| Training engine | Several paths | Newer Trainer plus old facade | REDESIGN | One Trainer core with strategy objects or config modes. |
| Checkpointing | Torch checkpoint dict | Training checkpoint directories | REDESIGN | Separate final model artifacts from resumable checkpoints. |
| Model loading | Native + HF/loaders | Richer loader/checkpoint packages | REDESIGN | Introduce `Model.load()` through registry/spec/artifact pipeline. |
| Model saving | `torch.save`, HF `save_pretrained` | Lifecycle artifacts | REDESIGN | `.arcmodel` begins as ArcLM-owned container using Torch tensors internally. |
| Artifact manifests | Limited metadata | Lifecycle/security manifests | PORT | Versioned manifest is mandatory from the first vNext artifact. |
| Device detection | Scattered | Runtime diagnostics | REWRITE | Add ArcLM-owned `Runtime.auto()` and keep backend objects internal. |
| Runtime precision | Scattered dtype strings | Precision helper | PORT | First slice reports supported dtype, later integrate with training. |
| Supported models | Static support matrix | Capability matrix | REDESIGN | Use capability-specific statuses, not boolean support. |
| External HF inference | Yes | Beta | WRAP | Keep optional integration, but do not make HF semantics the model API. |
| Security | Basic loading policy | Rich security core | PORT LATER | Use for artifact integrity/loading after core shape stabilizes. |
| Web Studio | No | Rich | DEFER | Product surface is not part of first core migration. |
| Agents/tools | No | Rich | DEFER | Outside LM framework foundation. |
| Distributed/FSDP | No | Experimental | INVESTIGATE | Defer until single-device core is coherent. |
| Hyperparameter search | Basic tracking only | Search module | DEFER | Useful for experiments after Trainer core. |
| Logic module | Yes | Yes | INVESTIGATE | Preserve temporarily; unclear relationship to LM vNext core. |

## Reuse / Wrap / Migrate / Deprecate

| Existing Area | Classification | Reason |
| --- | --- | --- |
| `arclm.model.ArcLM` | REUSE | Simple native causal LM, good first architecture. |
| `arclm.tokenizer.Tokenizer` | REUSE | Good enough for native vertical slice. |
| `arclm.trainer.Trainer` | WRAP THEN MIGRATE | Works and tested; needs runtime/artifact boundaries later. |
| `arclm.pipeline.train_model` | DEPRECATE LATER | Useful compatibility facade, but too much orchestration in one function. |
| `arclm.external_inference` | WRAP | Valuable behavior, but model loading/save semantics are too HF-shaped. |
| `arclm.sft` | WRAP THEN MIGRATE | Keep data/collator behavior; fold execution into unified Trainer later. |
| `arclm.resources` | MIGRATE | Existing device validation should feed new `Runtime`. |
| `arclm.loaders` | MIGRATE | Useful inspection/loading logic; should move under `Model.load()` pipeline. |
| `arclm.registry.py` | DEPRECATE LATER | File blocks target `arclm.registry` package; keep until compatibility plan exists. |
| `arclm/api.py` | DEPRECATE LATER | File blocks target `arclm.api` package; use transitional `arclm.vnext`. |

## First Vertical Slice Proposal

Based on the real repository, the first slice should be:

```text
Word-tokenized text
-> build/reuse Tokenizer
-> Model.create(architecture="arclm-native")
-> Runtime.auto()
-> deterministic generate
-> save .arcmodel directory with manifest/config/tokenizer/weights
-> load .arcmodel
-> deterministic generate again
-> compare config, weights, and output
```

Training is intentionally not rewritten in this first code patch. The existing
training suite already passes, and the new artifact/model/runtime boundary can
be proven with focused tests before the Trainer is routed through it.
