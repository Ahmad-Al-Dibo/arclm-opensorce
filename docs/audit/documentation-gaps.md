# Documentation Gap Analysis

## Fixed In This Update

- Replaced the web-interface-only README with a framework overview.
- Added a documentation website structure using MkDocs Material.
- Added a supported-models matrix with explicit support levels.
- Added quick-start, data, model, training, evaluation, inference, API, CLI, migration, production-readiness, and roadmap pages.
- Added public support metadata in `arclm.supported_models`.
- Added docstrings for `PreprocessConfig` and `PreprocessPipeline`.
- Updated examples README version language.

## Still Missing

- Complete generated API docs from docstrings.
- Full parameter/return/raises docstrings for every helper method.
- Automated runnable example tests for all examples.
- External model verification tests with controlled downloads/cache.
- Formal dataset schema documentation because the schema API does not exist yet.
- Release documentation for publishing docs.
- Dedicated security page for untrusted checkpoint handling.

## Inaccuracies Found

- README badge/text referenced `0.6.0` while the code is `0.6.1`.
- README positioned the project mainly as a simple web interface.
- Examples README said examples target `0.5.0`.
- Older docs in `docs/versions/0.5.0` are archival and should not be used as current 0.6.1 docs.
- CLI docs mention `arclm` command but packaging lacked a console-script entry point before this update.

