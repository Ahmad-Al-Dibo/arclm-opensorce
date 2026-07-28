# Production Operations Guide

ArcLM is intended for simple, auditable production-oriented workflows in the
open-source edition. Keep deployments focused on verified causal-language-model
paths and collect the structured reports ArcLM emits.

Recommended operational practices:

- install into a clean Python `3.9` to `3.12` environment
- run `arclm doctor --json` before long workflows
- pin model revisions for external Hugging Face models
- keep cache and run directories on writable local storage
- keep `trust_remote_code=False` unless the model repository is trusted
- inspect checkpoints with `arclm checkpoint inspect`
- verify v1 checkpoints with `arclm checkpoint verify`
- keep generated runs, caches, datasets, and weights out of source releases
- collect JSON reports from run directories for audit trails
- use `arclm config validate` in CI before running workflows

GPU support is runtime-detected. On CPU-only hosts, GPU certification is
untested and must not be claimed.
