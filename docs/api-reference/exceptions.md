# Exceptions

ArcLM mostly raises standard exceptions:

- `ValueError` for unsupported modes, tokenizers, config fields, and invalid generation options.
- `FileNotFoundError` for missing files.
- `ImportError` for missing optional dependencies such as PEFT or Transformers.
- `RuntimeError` for model-loading, generation, or checkpoint adaptation failures.

The `arclm.logics` package includes `EvaluationException`, but the logic utilities are unrelated to the primary causal-LM framework scope and should be treated as legacy/auxiliary.

ArcLM-specific exceptions:

- `ArcLMError`
- `ConfigurationError`
- `DatasetError`
- `DatasetValidationError`
- `DatasetFormatError`
- `ModelError`
- `UnsupportedModelError`
- `ModelLoadError`
- `ModelCompatibilityError`
- `TrainingError`
- `CheckpointError`
- `OptionalDependencyError`
