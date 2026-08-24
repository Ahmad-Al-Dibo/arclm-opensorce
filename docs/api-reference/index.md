# API Reference

This reference documents the ArcLM `1.0.0` public API: the ArcLM-first
framework layer, the lower-level data/model/training helpers, the CLI-facing
workflow pieces, and compatibility routes.

Start with:

- [Framework API](framework-api.md) for `Lab`, `Dataset`, `Model`, `Trainer`,
  `Runtime`, and `ModelRegistry`.
- [Full Public API](full-public-api.md) for the broad `arclm.__all__` export
  inventory and public methods found on exported classes.

Focused pages:

- [Data API](data-api.md)
- [Tokenization API](tokenization-api.md)
- [Model API](model-api.md)
- [Training API](training-api.md)
- [Evaluation API](evaluation-api.md)
- [Inference API](inference-api.md)
- [Configuration API](configuration-api.md)
- [Exceptions](exceptions.md)

Stability labels:

- Stable-ish: core behavior covered by local tests and examples.
- Experimental: usable but needs more validation or design hardening.
- Legacy: retained for backward compatibility.
- Internal: not intended as primary public API.
