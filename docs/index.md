# ArcLM

ArcLM is an ArcLM-first, self-driving framework for causal language-model workflows.

ArcLM is its own framework layer. PyTorch, Hugging Face, and related tools are backends or dependencies where they are useful; the public route belongs to ArcLM. Its foundations include schema validation, composable data pipelines, streaming dataset sources, tokenization caching, native ArcLM causal-language-model training and inference, typed workflow configuration, safe checkpoint inspection, `.arcmodel` artifacts, and documented Hugging Face causal-LM paths.

## Workflow

```text
Raw data -> Loading -> Cleaning -> Validation -> Transformation -> Formatting
-> Tokenization -> Model loading -> Training or fine-tuning -> Evaluation
-> Inference -> Reporting
```

Start with [Installation](installation.md), then follow the [Quick Start](quick-start.md).

## Key Pages

- [Project Vision](project-vision.md)
- [Framework API](api-reference/framework-api.md)
- [Full Public API](api-reference/full-public-api.md)
- [Supported Models](supported-models.md)
- [Data Guide](data-guide/loading-data.md)
- [Data at Scale](data-at-scale.md)
- [Workflow Runner](workflow-runner.md)
- [Model Loading Guide](model-guide/loading-models.md)
- [Training Guide](training-guide/training-configuration.md)
- [API Reference](api-reference/index.md)
- [Operational Readiness](production-readiness.md)
- [Roadmap](roadmap.md)
