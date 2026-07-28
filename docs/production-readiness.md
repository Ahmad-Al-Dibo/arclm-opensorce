# Operational Readiness

ArcLM is designed as a simple, production-oriented open-source framework for data-first causal-language-model workflows. This page describes the guarantees that are already implemented and the boundaries users should account for when adopting the library.

| Area | Status | Notes |
| --- | --- | --- |
| API stability | Ready for focused APIs | Stable APIs are inventoried in `arclm.stability` and covered by a snapshot fixture. |
| Configuration validation | Ready for workflow configs | `ArcLMConfig` provides typed schema-versioned configuration, strict unknown-field handling, path normalization, and migration reports. |
| Logging | Partially ready | Package loggers and CLI verbosity exist; some legacy training paths still emit progress directly. |
| Error handling | Partially ready | Structured exceptions and actionable model/checkpoint/config errors exist; older paths continue to be hardened. |
| Type safety | Partially ready | New stable hardening modules run under stricter MyPy settings; legacy modules remain staged. |
| Dependency management | Partially ready | Extras exist; docs deps and CLI entry point were added. |
| Deterministic behavior | Partially ready | Seeds exist in key paths; full reproducibility is not guaranteed. |
| Dataset validation | Partially ready | Formal record schemas and reports exist; richer schemas and streaming validation remain. |
| Model compatibility checks | Partially ready | Native tokenizer checks and runtime support inspection exist; tiny GPT-2 is certified, broader families are not. |
| Checkpoint reliability | Partially ready | Native checkpoints save tokenizer metadata; resume semantics need more tests. |
| Testing | Partially ready | Core, CLI, integration, docs, security, checkpoint, scale, and CPU certification tests are available. |
| Documentation coverage | Partially ready | MkDocs documents the focused workflow, APIs, CLI, support levels, and operational guidance. |
| Security | Partially ready | Safe checkpoint inspection, explicit loading policies, secret scanning, and `trust_remote_code=False` defaults are in place. |
| Large datasets | Experimental | Main `DataProcessor` is in-memory; preprocessing streams JSONL writes but uses simple heuristics. |
| CPU/GPU behavior | Partially ready | Device options exist; CI/device matrix is missing. |
| Packaging | Partially ready | Console script and docs extras are configured; old MANIFEST references need release validation. |
| Semantic versioning | Partially ready | Version is present; migration/deprecation policy needs enforcement. |
| Release process | Partially ready | `setup.py release` exists; CI and docs publishing need definition. |

## Hardening Roadmap

1. Consolidate CLI training on `train_model`.
2. Add schema validation primitives for datasets.
3. Add automated docs build and link checks in CI.
4. Add Python 3.9-3.12 test matrix with valid Torch wheels.
5. Certify one external model family with automated loading, tokenizer, inference, and SFT smoke tests.
6. Define public API stability and deprecation policy.
7. Remove or deprecate unrelated top-level exports after migration guidance.
