# Contributing

Contributions should keep ArcLM focused on data-first causal-language-model workflows.

## Expectations

- Inspect existing APIs before adding new ones.
- Add tests or reproducible examples for new public behavior.
- Do not claim official model support without verification.
- Document optional dependencies and hardware requirements.
- Keep examples small and runnable.
- Preserve backward compatibility or provide migration notes.

## Local Checks

```bash
python -m pytest tests
mkdocs build --strict
```

