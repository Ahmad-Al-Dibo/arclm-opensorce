# ArcLM vNext Implementation Status

| Component | Status | Tests | Notes |
| --- | --- | --- | --- |
| Architecture audit | PARTIAL | N/A | Open source and special repo inspected. |
| Feature inventory | PARTIAL | N/A | Initial decisions recorded. |
| Baseline | COMPLETE | PASS | Project venv: 87 passed, 1 skipped. |
| Runtime | IMPLEMENTED | PASS | CPU/CUDA inspection through Torch, ArcLM-owned public object. |
| ModelSpec/Registry | IMPLEMENTED | PASS | Native causal LM spec only. |
| Model wrapper | PARTIAL | PASS | Create/load/save/generate/inspect for native ArcLM only. |
| `.arcmodel` artifact | EXPERIMENTAL | PASS | Single-file zip, directory, and one-shard sharded layouts with manifest and hashes. |
| Unified Trainer routing | PARTIAL | PASS | Public `Trainer(model, dataset)` delegates to existing tested Trainer loop; legacy constructor still bridges. |
| Dataset inspect/prepare vNext | PARTIAL | PASS | Public `Dataset.load/inspect/prepare` supports local TXT/JSON/JSONL and word-tokenizer next-token batches. |
| Lab | PARTIAL | PASS | `Lab.inspect/create/plan/train/pretrain` delegates to Dataset, Model, Runtime, Trainer, and Artifact layers. |

## Verification

Post-change command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
96 passed, 1 skipped in 99.35s (0:01:39)
```

Current migration-phase focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_vnext_core.py -q
```

Result:

```text
9 passed in 21.26s
```
