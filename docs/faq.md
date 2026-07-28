# Frequently Asked Questions

## Is ArcLM production-ready?

No. ArcLM `0.8.0.dev0` is a pre-1.0 framework. See [Production Readiness](production-readiness.md).

## Does ArcLM support every Hugging Face model?

No. ArcLM targets causal language models. Generic Hugging Face loading uses `AutoModelForCausalLM`, but official support requires ArcLM-specific verification.

## Can I use BERT or T5?

Not through the current public workflow. Encoder-only and seq2seq models are not supported in `0.8.0.dev0`.

## What should I use first?

Use `DataProcessor` for small in-memory data preparation, `PreprocessPipeline` for JSONL cleaning reports, and `train_model` or `load_any_model` for causal-LM workflows.
