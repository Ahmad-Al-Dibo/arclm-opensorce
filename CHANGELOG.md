# Changelog

## 1.0.0

- Rewrote the README around ArcLM's new self-driving framework direction:
  a connected route from raw data to validation, tokenization, model loading,
  training or fine-tuning, evaluation, artifact saving, and reporting.
- Clarified that ArcLM is positioned as its own framework, with PyTorch and
  Hugging Face treated as dependencies or backends instead of the project
  identity.
- Updated package metadata for the ArcLM-first framework direction and promoted
  the package version to `1.0.0`.
- Shortened the public project pitch and aligned it with the current
  workflow runner, typed configuration, checkpoint safety, diagnostics,
  native artifact, and Hugging Face causal-LM surfaces.
- Expanded the docs site with a Framework API page and a generated full public
  API inventory covering exported functions, classes, and public methods.

## 0.9.0

- Added a central public API stability manifest and stable API snapshot tests.
- Added schema-versioned typed ArcLM workflow configuration with strict parsing,
  explicit environment expansion, redacted effective config export, and
  migration reporting for legacy field names.
- Added formal checkpoint manifest inspection, safe/trusted/legacy loading
  policies, hash verification, and checkpoint CLI commands.
- Added explicit device and precision validation and an `arclm doctor` command.
- Added model-certification protocol scaffolding, release artifact checksums,
  package-content scanning, and SBOM generation helpers.
- Added experimental CPU certification for the tiny random Llama-compatible
  Hugging Face test artifact.
- Added Phase 4 hardening tests for configuration, checkpoint security,
  API compatibility, diagnostics, and CLI surfaces.

## 0.8.0.dev0

- Added streaming dataset sources, deterministic sharding, deterministic
  splitting, exact duplicate detection, leakage checks, and privacy-aware data
  quality reports.
- Added schema-aware tokenization workflows with deterministic JSON cache keys.
- Added reproducibility fingerprints, local run directories, workflow dry-run
  support, callback events, evaluation reports, batched generation reports,
  extension registries, security helpers, and benchmark smoke utilities.
- Expanded CLI commands for data analysis/splitting/sharding/fingerprinting,
  runs, cache, plugins, and workflow execution.
- Added staged Ruff and MyPy configuration and Phase 3 tests.

## 0.7.0.dev0

- Reframed ArcLM as a data-first framework for causal-language-model workflows.
- Added MkDocs documentation structure and production-readiness reporting.
- Added explicit model-support metadata and support-level definitions.
- Added docs extras and an `arclm` console script entry point.
- Documented preprocessing public APIs.
- Added formal dataset schemas, validation reports, composable data pipelines,
  model support inspection, a consolidated model facade, CLI consolidation,
  checkpoint inspection helpers, structured exceptions, logging helpers, and
  deprecation utilities.

## 0.6.1

- Current repository release.
