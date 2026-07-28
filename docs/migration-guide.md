# Migration Guide

ArcLM `0.9.0` keeps existing public APIs in place while adding release-candidate hardening APIs.

## Changed Positioning

ArcLM should now be described as a data-first framework for causal language models, not a broad machine-learning toolkit.

## Preferred Imports

Use these public imports:

```python
from arclm import DataProcessor, Tokenizer, train_model, load_model
from arclm import inspect_model_source, load_any_model, train_sft
from arclm.preprocess import PreprocessConfig, PreprocessPipeline
```

## Deprecated Or Legacy Areas To Plan For

No APIs are removed in this update, but future releases should consider deprecating or moving:

- `MiniGPT`: keep as an alias for `ArcLM` until a formal deprecation window exists.
- `MiniGPT`: instantiating this alias now emits `DeprecationWarning`; use `ArcLM`.
- `pipeline_v2`: currently preserves legacy helper imports.
- `checkpoint_is_compatible_for_tuining`: typo retained for compatibility and emits `DeprecationWarning`; use `checkpoint_is_compatible_for_tuning`.
- Top-level `logics` exports: unrelated to data-first causal-LM workflows.
- Older CLI internals that use `UnifiedPipeline` instead of `train_model`.

The `logics` objects remain importable from `arclm` for compatibility but are no longer included in the primary `arclm.__all__` public API list.

## Version Recommendation

The repository should use development version `0.9.0` and plan the first release candidate as `0.9.0rc1`; the `1.0.0` line should wait until the compatibility gates in the release checklist are complete.
