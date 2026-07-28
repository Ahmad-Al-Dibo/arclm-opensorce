# Project Vision

ArcLM is a focused Python framework for preparing language-model data and building reproducible workflows for causal language models.

## Who It Is For

ArcLM is for Python developers who want to prepare text or instruction data, tokenize it, run compact native causal-LM training, or connect carefully to Hugging Face causal-language-model workflows.

## Problems It Solves

- Repeated dataset loading and formatting code.
- Inconsistent instruction and conversation formatting.
- Tokenizer/checkpoint mismatches.
- Unclear model-loading behavior.
- Documentation that overclaims model compatibility.

## Why Data Preparation Comes First

Model behavior is strongly shaped by the data pipeline. ArcLM treats loading, cleaning, validation, formatting, and tokenization as first-class framework stages instead of incidental script code.

## Why Causal Language Models First

The implemented native model is decoder-only and causal. The Hugging Face paths call `AutoModelForCausalLM`. ArcLM therefore focuses on causal language modeling instead of pretending to support every ML architecture.

## Non-Goals

- Replacing Transformers, PyTorch Lightning, Accelerate, or datasets platforms.
- Supporting encoder-only or seq2seq models in the current public workflow.
- Claiming official support for untested model families.
- Hiding missing validation or production-readiness gaps.

## Design Principles

- Data-first workflows.
- Explicit configuration.
- Reproducibility.
- Clear validation.
- Small composable components.
- Honest model support.
- Useful defaults without hidden behavior.
- Framework independence where practical.
- Stable public APIs.
- Strong documentation.

## Future Expansion

ArcLM can expand by adding verified causal-LM families, stronger dataset schemas, richer evaluation, and production validation. New capabilities should fit the data-to-causal-LM workflow and include tests or reproducible examples.

