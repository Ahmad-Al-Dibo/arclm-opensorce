# Troubleshooting

## `ModuleNotFoundError: torch.utils`

Check Python and Torch compatibility. ArcLM declares Python `>=3.9,<3.13`; unsupported Python versions can have incomplete or incompatible Torch installations.

## Hugging Face Model Cannot Load

Install optional dependencies:

```bash
pip install -e ".[hf]"
```

Then inspect the source first:

```python
from arclm import inspect_model_source

print(inspect_model_source("gpt2").format_report())
```

## Tokenizer Mismatch

For native checkpoints, continue training requires matching tokenizer metadata. Use checkpoints saved by `train_model()` or `Trainer.save()`.

## Preprocessing Drops Too Many Rows

Relax `PreprocessConfig` thresholds such as `min_chars`, `min_words`, `min_entropy`, `allowed_languages`, or dedup settings, then inspect the generated report.

