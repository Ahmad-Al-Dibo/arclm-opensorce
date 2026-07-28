# ArcLM

ArcLM is a focused Python framework for preparing language-model data and building reproducible workflows for causal language models.

ArcLM `0.8.0.dev0` is a pre-1.0 project. Its reliable foundations include schema validation, composable data pipelines, streaming dataset sources, tokenization caching, native ArcLM causal-language-model training/inference, and a certified tiny GPT-2 Hugging Face workflow.

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
- [Production Readiness](production-readiness.md)
- [Roadmap](roadmap.md)
