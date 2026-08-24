# ArcLM vNext Implementation Status

| Component | Status | Tests | Notes |
| --- | --- | --- | --- |
| Architecture audit | PARTIAL | N/A | Open source and special repo inspected. |
| Feature inventory | PARTIAL | N/A | Initial decisions recorded. |
| Baseline | COMPLETE | PASS | Previous project venv baseline: 96 passed, 1 skipped. |
| Runtime | IMPLEMENTED | PASS | CPU/CUDA inspection through Torch, ArcLM-owned public object. |
| Core contracts | IMPLEMENTED | PASS | Protocol contracts exist for model/spec/registry/runtime/dataset/backend/training/artifact/weight IO/adapter. |
| ModelSpec/Registry | IMPLEMENTED | PASS | Native causal LM spec includes builder, requirements, capability labels, weight mapper and adapter targets. |
| Model wrapper | PARTIAL | PASS | Create/load/save/generate/inspect and native LoRA attach/save/load for ArcLM native family. |
| Training Engine | IMPLEMENTED | PASS | Public high-level Trainer uses `TrainingEngine` with pretrain/full-finetune/adapter strategies. |
| `.arcmodel` artifact | IMPLEMENTED | PASS | Single zip, directory and true multi-shard safetensors layouts with manifest/index/hash/tensor validation. |
| `.arcadapter` artifact | IMPLEMENTED | PASS | Native LoRA adapter manifest, safetensors tensors and base fingerprint compatibility check. |
| Unified Trainer routing | PARTIAL | PASS | High-level Trainer no longer delegates to legacy. Legacy positional constructor still bridges. |
| Dataset inspect/prepare vNext | PARTIAL | PASS | Public `Dataset.load/inspect/prepare` supports local TXT/JSON/JSONL and word-tokenizer next-token batches. |
| Lab | PARTIAL | PASS | `Lab.inspect/create/plan/train/pretrain` delegates to Dataset, Model, Runtime, Trainer, and Artifact layers. |
| External pretrained loaders | TRANSITIONAL | Existing | Transformers/PEFT loaders remain outside the new Core path. |

## Legacy Dependency Register

| Dependency | Status | Location |
| --- | --- | --- |
| old Trainer behind high-level vNext Trainer | REMOVED | `arclm.vnext.Trainer(model, dataset)` now uses `arclm.core.TrainingEngine`. |
| old Trainer positional constructor | TRANSITIONAL | `Trainer(raw_model, optimizer, criterion, config)` still bridges for compatibility. |
| Transformers loaders | TRANSITIONAL | `arclm.models`, `arclm.external_inference`, `arclm.loaders.hf_loader`, SFT helpers and examples. |
| PEFT orchestration | TRANSITIONAL | Legacy SFT/external adapter paths only. Native LoRA does not require PEFT. |
| Torch serialization in `.arcmodel` | REMOVED | Native `.arcmodel` uses safetensors. |
| Torch serialization in legacy checkpoints/loaders | ISOLATED | `arclm.trainer`, `arclm.pipeline`, `arclm.checkpoints`, `arclm.inference`, `arclm.training.unified`, external loaders. |

## Verification

Post-change command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
102 passed, 1 skipped in 235.49s (0:03:55)
```

Current migration-phase focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vnext_core.py tests\test_vnext_engine_artifact_adapter.py -q
```

Result:

```text
15 passed in 32.09s
```

## Performance Smoke

Environment: CPU runtime, tiny `arclm-native-causal-lm` model.

| Metric | Value |
| --- | ---: |
| Load time | 0.072538 s |
| Python peak RAM during load | 0.187 MB |
| Peak VRAM | 0.000 MB |
| Generation smoke speed | 8 tokens in 0.017768 s |
| `.arcmodel` size | 5,754 bytes |
| `.arcadapter` size | 2,792 bytes |
