# Changelog

## 1.0.0

- Reframed the README around ArcLM's self-driving route: inspect, validate,
  prepare, tokenize, load, train or fine-tune, evaluate, save, and report.
- Clarified the public positioning: ArcLM is its own framework, with PyTorch
  and Hugging Face treated as dependencies or backends.
- Updated package metadata and promoted the package version to `1.0.0`.
- Shortened the public project overview so the framework direction is clearer
  for new users.
- Expanded the docs site with a Framework API page and a generated full public
  API inventory covering exported functions, classes, and public methods.

## 0.9.0

- Added API stability manifest and snapshot tests.
- Added typed configuration schema, migration reporting, doctor diagnostics,
  safe checkpoint inspection, trust policies, and release-candidate helpers.
- Added experimental CPU certification for `hf-internal-testing/tiny-random-LlamaForCausalLM`.

## 0.8.0.dev0

- Added streaming dataset sources, deterministic sharding/splitting,
  duplicate/leakage checks, data-quality analysis, tokenization caching,
  fingerprints, run directories, workflow dry-runs, evaluation reports,
  batched generation, registries, security helpers, benchmarks, and expanded
  CLI commands.

## 0.7.0.dev0

- Reframed ArcLM as a data-first framework for causal-language-model workflows.
- Added a MkDocs documentation site.
- Added explicit supported-model levels and a `ModelCapability` representation.
- Added docs for getting started, data preparation, model loading, training, evaluation, inference, CLI, configuration, migration, and production readiness.
- Added docs extras and an `arclm` console script entry point in packaging metadata.
- Documented preprocessing public APIs.
- Added formal dataset schemas, validation reports, composable data pipelines,
  model support inspection, a consolidated model facade, CLI consolidation,
  checkpoint inspection helpers, structured exceptions, logging helpers, and
  deprecation utilities.

## 0.6.1

- Current repository version.
