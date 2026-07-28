# Contributing To ArcLM

ArcLM contributions should keep the project focused on data-first workflows for causal language models.

Before opening a change:

- Run `python -m pytest tests` in a supported Python version (`>=3.9,<3.13`).
- Run `mkdocs build --strict` when changing docs.
- Add tests or reproducible examples for new public behavior.
- Do not claim official model support without loading, tokenizer, inference, and training/fine-tuning verification where claimed.
- Preserve backward compatibility or document a migration path.

