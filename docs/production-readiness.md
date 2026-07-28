# Production Readiness

ArcLM `0.8.0.dev0` should not be described as production-ready. It is a useful pre-1.0 framework with stronger foundations and clear gaps.

| Area | Status | Notes |
| --- | --- | --- |
| API stability | Partially ready | Many top-level exports exist; formal stability/deprecation policy is missing. |
| Configuration validation | Partially ready | `create_config` rejects unknown fields, but `Config(**kwargs)` accepts unknown/unused values indirectly. |
| Logging | Partially ready | Trainer metrics logging exists; several code paths still print directly. |
| Error handling | Partially ready | External loading has helpful errors; older CLI paths and configs need cleanup. |
| Type safety | Experimental | Some annotations exist; many methods remain untyped. |
| Dependency management | Partially ready | Extras exist; docs deps and CLI entry point were added. |
| Deterministic behavior | Partially ready | Seeds exist in key paths; full reproducibility is not guaranteed. |
| Dataset validation | Partially ready | Formal record schemas and reports exist; richer schemas and streaming validation remain. |
| Model compatibility checks | Partially ready | Native tokenizer checks and runtime support inspection exist; tiny GPT-2 is certified, broader families are not. |
| Checkpoint reliability | Partially ready | Native checkpoints save tokenizer metadata; resume semantics need more tests. |
| Testing | Partially ready | Unit tests cover core paths; environment here could not run due unsupported Python/Torch. |
| Documentation coverage | Partially ready | MkDocs site now documents current scope and gaps. |
| Security | Experimental | `torch.load(weights_only=False)` is documented as trusted-only, but broader security guidance is limited. |
| Large datasets | Experimental | Main `DataProcessor` is in-memory; preprocessing streams JSONL writes but uses simple heuristics. |
| CPU/GPU behavior | Partially ready | Device options exist; CI/device matrix is missing. |
| Packaging | Partially ready | Console script and docs extras are configured; old MANIFEST references need release validation. |
| Semantic versioning | Partially ready | Version is present; migration/deprecation policy needs enforcement. |
| Release process | Partially ready | `setup.py release` exists; CI and docs publishing need definition. |

## Path To A Stable Release

1. Consolidate CLI training on `train_model`.
2. Add schema validation primitives for datasets.
3. Add automated docs build and link checks in CI.
4. Add Python 3.9-3.12 test matrix with valid Torch wheels.
5. Certify one external model family with automated loading, tokenizer, inference, and SFT smoke tests.
6. Define public API stability and deprecation policy.
7. Remove or deprecate unrelated top-level exports after migration guidance.
