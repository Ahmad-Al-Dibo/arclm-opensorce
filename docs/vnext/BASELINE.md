# ArcLM vNext Baseline

Baseline date: 2026-08-23.

## Environment

| Item | Value |
| --- | --- |
| Repository | `D:\AhmadAlDibo-WORKSPACE\2026\Projects\ArcLM-LLMs-Library\ArcLM-opensorce\ArcLM-opensorce` |
| System Python | `3.14.6` |
| Project venv Python | `3.12.10` |
| Supported Python range | `>=3.9,<3.13` |
| OS | Windows 11 `10.0.22631` |
| Torch | `2.13.0+cpu` |
| Transformers | `5.14.1` |
| SentencePiece | `0.2.2` |
| PEFT | `0.19.1` |
| Safetensors | `0.8.0` |
| Pytest | `9.1.1` |
| CUDA available | `False` |
| CUDA device count | `0` |

## Test Runs

### System Python

Command:

```powershell
python -m pytest -q
```

Result: FAILED during collection.

Known reason: the system interpreter is Python `3.14.6`, outside the declared
support range, and the visible Torch installation is incomplete for this
project (`ModuleNotFoundError: No module named 'torch.utils'`).

### Project Virtual Environment

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Result:

```text
87 passed, 1 skipped in 277.71s (0:04:37)
```

This venv is the baseline used for vNext verification.

## Known Baseline Risks

- CPU-only environment; CUDA, ROCm, MPS and multi-GPU behavior are not validated.
- Direct Torch import took noticeably longer than simple package metadata
  inspection, but completed successfully.
- Existing worktree had unrelated uncommitted edits before vNext work:
  `arclm/cli.py`, `arclm/preprocess/pipeline.py`,
  `arclm/preprocess/report.py`, `examples/03_data_processing.py`,
  `examples/10_preprocess_pipeline.py`, and
  `examples/15_custom_hf_sft_loop.py`.
