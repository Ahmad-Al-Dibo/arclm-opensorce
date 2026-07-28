# ArcLM

ArcLM is a focused Python framework for preparing language-model data and building reproducible workflows for causal language models.

ArcLM is the open-source edition of a simple, production-oriented framework for data-first causal-language-model workflows. Its foundations include schema validation, composable data pipelines, streaming dataset sources, tokenization caching, native ArcLM causal-language-model training/inference, typed workflow configuration, safe checkpoint inspection, and certified tiny GPT-2 Hugging Face workflow tests.

## Workflow

```text
Raw data -> Loading -> Cleaning -> Validation -> Transformation -> Formatting
-> Tokenization -> Model loading -> Training or fine-tuning -> Evaluation
-> Inference -> Reporting
```

Start with [Installation](installation.md), then follow the [Quick Start](quick-start.md).

## Key Pages

- [Project Vision](project-vision.md)
- [Supported Models](supported-models.md)
- [Data Guide](data-guide/loading-data.md)
- [Data at Scale](data-at-scale.md)
- [Workflow Runner](workflow-runner.md)
- [Model Loading Guide](model-guide/loading-models.md)
- [Training Guide](training-guide/training-configuration.md)
- [API Reference](api-reference/index.md)
- [Operational Readiness](production-readiness.md)
- [Roadmap](roadmap.md)
