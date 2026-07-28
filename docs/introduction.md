# Introduction

ArcLM provides composable Python APIs for dataset loading, cleaning, formatting, tokenization, native causal-LM training, model loading, evaluation, and inference.

The canonical package name is `arclm`; the project name used in user-facing text is ArcLM. Version `0.8.0.dev0` is pre-1.0 and should be adopted with the current limitations in mind.

## What Exists Today

- `DataProcessor` loads JSON, JSONL, CSV, TXT, or custom loaders into an in-memory `ProcessedDataset`.
- `validate_records` validates text, prompt-completion, instruction, and conversation records.
- `DataPipeline` composes deterministic in-memory data operations with structured reports.
- `PreprocessPipeline` cleans and filters JSONL datasets with reports.
- `Tokenizer` and `SentencePieceTokenizer` support word and SentencePiece tokenization.
- `ArcLM` is a compact GPT-style causal LM implemented in PyTorch.
- `train_model` trains, fine-tunes, or continues native ArcLM checkpoints.
- `load_model` loads native ArcLM checkpoints for inference.
- `load_any_model`, `train_sft`, and related helpers integrate with Hugging Face causal-LM workflows.

## What Does Not Exist Yet

- Full production-readiness guarantees.
- Official support for every model that Transformers can load.
- General support for encoder-only, seq2seq, vision, audio, or multimodal models.
- A formal API deprecation policy.
