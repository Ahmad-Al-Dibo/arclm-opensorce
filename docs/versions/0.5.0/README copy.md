# ArcLM Documentation

This directory contains the deeper documentation for ArcLM 0.5.0. The root `README.md` is the public landing page for GitHub and PyPI; these files expand the same implemented workflows.

## Start Here

- [Usage Guide](USAGE.md): practical training, SFT, preprocessing, inference, and troubleshooting workflows.
- [Full Feature Guide](FULL_FEATURE_GUIDE.md): broader tour of the package surface and how modules fit together.
- [API Reference](reference/LIBRARY_README.md): public imports grouped by workflow.
- [Custom Trainers](CUSTOM_TRAINER.md): lower-level `Trainer` usage and custom training patterns.
- [Release Guide](VERSIONING.md): version source of truth and release checklist.

## Current Public APIs

| Goal | API |
| --- | --- |
| Train a native model from text | `train_model(mode="pretrain", ...)` |
| Fine-tune a native/adapted checkpoint | `train_model(mode="finetune", ...)` |
| Continue compatible native training | `train_model(mode="continue_training", ...)` |
| Run Hugging Face SFT | `train_sft(backend="huggingface", ...)` |
| Train native assistant-only examples | `InstructionDataset` + `Trainer` |
| Load a native checkpoint | `load_model(...)` |
| Inspect external sources | `SmartLoader.inspect(...)` |
| Prepare records | `DataProcessor.load(...).clean().transform(...)` |

## Boundaries In 0.5.0

- Hosted provider clients for OpenAI, Anthropic, Google Gemini, and Ollama are not implemented.
- Preference training such as DPO, RLHF, PPO, and reward modeling is not implemented.
- Native ArcLM LoRA layers are not implemented; LoRA support is available through PEFT for Hugging Face SFT.
- Native `ArcLM.forward()` accepts token IDs only. Hugging Face SFT uses attention masks through Transformers.
- The Python APIs are the canonical public workflows.

When code changes, update the matching docs and examples in the same change.
