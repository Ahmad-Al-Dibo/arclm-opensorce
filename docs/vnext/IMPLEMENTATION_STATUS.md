# ArcLM vNext Implementation Status

| Component | Status | Tests | Notes |
| --- | --- | --- | --- |
| Architecture audit | PARTIAL | N/A | Open source and special repo inspected. |
| Feature inventory | PARTIAL | N/A | Initial decisions recorded. |
| Baseline | COMPLETE | PASS | Project venv: 87 passed, 1 skipped. |
| Runtime | IMPLEMENTED | PASS | CPU/CUDA inspection through Torch, ArcLM-owned public object. |
| ModelSpec/Registry | IMPLEMENTED | PASS | Native causal LM spec only. |
| Model wrapper | PARTIAL | PASS | Create/load/save/generate/inspect for native ArcLM only. |
| `.arcmodel` artifact | EXPERIMENTAL | PASS | Directory layout with versioned manifest and hashes. |
| Unified Trainer routing | PLANNED | N/A | Existing Trainer unchanged. |
| Dataset inspect/prepare vNext | PLANNED | N/A | Existing data utilities unchanged. |
| Lab | PLANNED | N/A | Wait until core vertical slice includes training. |

## Verification

Post-change command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
90 passed, 1 skipped in 310.34s (0:05:10)
```
