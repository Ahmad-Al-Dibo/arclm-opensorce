# ArcLM Library Reference

This reference lists the current public API and the intended use of each major
module. For runnable workflows, see `../USAGE.md`.

ArcLM 0.5.0 supports native ArcLM checkpoints and Hugging Face SFT workflows.
Hosted provider clients such as OpenAI, Anthropic, Google Gemini, and Ollama
are outside this release.

## Top-level Imports

The root package exports the main user-facing surface:

```python
from arclm import (
    Config,
    DataProcessor,
    InstructionDataset,
    Trainer,
    build_model,
    build_trainer,
    train_model,
    train_sft,
    load_model,
    load_external_model,
    adapt_for_training,
    tokenizer_from_checkpoint,
)
```

## Configuration

`Config` is a plain Python configuration object.

Common fields:

- model: `embed_dim`, `block_size`, `num_blocks`, `dropout`, `vocab_size`
- training: `batch_size`, `num_epochs`, `learning_rate`, `weight_decay`, `grad_clip`, `device`
- data: `data_path`, `domain_data_path`, `domain_data_repeats`, `validation_split`
- tokenizer: `tokenizer_type`, `max_vocab`, `user_defined_symbols`, SentencePiece settings
- fine-tuning: `freeze_backbone`, `freeze_embedding`, `use_discriminative_lr`, `lr_multiplier`
- checkpoints/logging: `model_path`, `checkpoint_interval`, `checkpoint_batch_interval`, `metrics_log_path`

Use `create_config()` when you want validation for unknown keys.

## Native Model

`ArcLM(vocab_size, embed_dim, block_size, num_blocks, dropout)` is the native
causal language model.

Current native model notes:

- It has a causal attention mask internally.
- It does not expose a public `attention_mask` argument.
- It does not have a public `num_heads` parameter.

## Tokenizers

`Tokenizer`

- Word-level tokenizer.
- Methods: `build`, `encode`, `encode_text`, `decode`, `decode_string`, `get_vocab_size`, `to_checkpoint`.

`SentencePieceTokenizer`

- Subword tokenizer with BPE or unigram modes.
- Stores serialized SentencePiece model data in checkpoint metadata.

`TokenizerFactory`

- `TokenizerFactory.create("word", ...)`
- `TokenizerFactory.create("sentencepiece", ...)`
- `TokenizerFactory.register(name, cls)`

## Data

`DataProcessor.load(path)` supports:

- `.txt`
- `.json`
- `.jsonl`
- `.csv`
- custom loader functions

`ProcessedDataset` supports:

- `.clean()`
- `.filter(predicate)`
- `.transform(...)`
- `.tokenize(tokenizer)`
- `.split(...)`
- `.map_batches(...)`

`prepare_data(config, existing_tokenizer=None)` returns `DataBundle` with
tokens, train/validation splits, tokenizer, encoded IDs, and DataLoaders.

## Native Training

`build_model(config, vocab_size=None)` creates native `ArcLM`.

`build_trainer(model, config)` creates `Trainer` with AdamW and cross-entropy.

`Trainer.train(...)` accepts:

- tuple/list batches: `(x, y)`
- dict batches: `{"x": x, "y": y, "mask": optional_loss_mask}`

Shape contract:

- `x`: token IDs shaped `[batch, time]`
- `y`: next-token labels shaped `[batch, time]`
- `mask`: optional float/bool loss mask shaped `[batch, time]`
- `model(x)`: logits shaped `[batch, time, vocab]`

Important methods:

- `train`
- `compute_loss`
- `unpack_batch`
- `freeze_layers`
- `unfreeze_layers`
- `get_frozen_layers_info`
- `save`
- `load`
- `get_train_history`

Extension points:

- override `compute_loss(...)` for custom objectives that still use ArcLM-shaped logits;
- override `unpack_batch(...)` for a different batch structure;
- write a manual loop around `unpack_batch(...)` and `compute_loss(...)` for custom schedulers, gradient accumulation, or extra logging;
- pass `checkpoint_callback`, `checkpoint_epoch_interval`, or `checkpoint_batch_interval` to `train(...)` for custom checkpoint timing.

When saving native checkpoints, include tokenizer metadata:

```python
trainer.save(
    config,
    vocab=tokenizer.vocab,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    tokenizer_metadata=tokenizer.to_checkpoint(),
)
```

See `../CUSTOM_TRAINER.md` for complete low-level examples.

## `train_model`

High-level native ArcLM workflow:

```python
train_model(
    mode="pretrain" | "finetune" | "continue_training",
    data="data/file.txt",
    output="models/model.pth",
    checkpoint=None,
    tokenizer=None,
    **config_overrides,
)
```

Modes:

- `pretrain`: new native ArcLM checkpoint from text.
- `finetune`: next-token fine-tune a checkpoint or external source.
- `continue_training`: strict compatible continuation.

Returns `TrainingResult`.

## `train_sft`

Public Hugging Face SFT workflow:

```python
from arclm import train_sft

result = train_sft(
    model="Qwen/Qwen3-0.6B",
    dataset="examples/qwen3_0_6b_sft/data/sample_sft.jsonl",
    output_dir="examples/qwen3_0_6b_sft/output/qwen3_0_6b_sft_lora",
    backend="huggingface",
    assistant_only_loss=True,
    use_lora=True,
)
```

Implemented options:

- `backend="huggingface"` only.
- `assistant_only_loss=True | False`.
- `use_lora=True | False`.
- `batch_size`, `gradient_accumulation_steps`, `learning_rate`, `num_epochs`.
- `max_length`, `dtype`, `device_map`.
- `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`.
- `max_steps` for smoke tests.

Returns `SFTTrainingResult`:

- `model`
- `dataset`
- `output_dir`
- `backend`
- `use_lora`
- `assistant_only_loss`
- `train_loss_history`
- `steps`
- `metadata_path`
- `adapter_path`
- `full_model_path`

If `use_lora=True`, install `peft`. If `use_lora=False`, a full Hugging Face
model is saved to `output_dir`.

## Instruction Dataset

`InstructionDataset(instructions, responses, tokenizer, block_size, ...)`
formats examples as:

```text
<|instruction|>
instruction
<|response|>
response
```

It returns `{"x", "y", "mask"}` where `mask` is aligned to next-token labels.
Assistant response labels are active; prompt and padding labels are ignored.

`create_instruction_dataloader(...)` wraps the dataset in a DataLoader.

## Checkpoint Loading

`load_model(path, device="cpu")`

- Loads native ArcLM checkpoints for inference.
- Returns `LoadedModel`.

`load_external_model(source, map_location="cpu")`

- Normalizes supported sources into `LoadedCheckpoint`.

`adapt_for_training(loaded, target_config=None, tokenizer=None, ...)`

- Builds native `ArcLM`.
- Copies compatible weights by matching tensor names and shapes.

`tokenizer_from_checkpoint(checkpoint_or_source)`

- Restores word or SentencePiece tokenizer metadata from native ArcLM checkpoints.

## SmartLoader

Use `SmartLoader.inspect(source)` before loading unknown or large sources.

It can identify:

- source type
- weight format
- tokenizer files
- adapter-like files
- optimizer/trainer state
- suggested loading plan

## Diagnostics

Common helpers:

- `calculate_metrics`
- `calculate_perplexity`
- `export_metrics_to_json`
- `export_metrics_to_markdown`
- `predict_top_k`
- `format_top_k_predictions`
- `format_tokenizer_coverage_report`
- `run_long_context_evaluation`
- `score_concept_relationships`

## Preprocessing

`arclm.preprocess` exports:

- `PreprocessConfig`
- `PreprocessPipeline`

Submodules include:

- `cleaner`
- `filters`
- `duplicate`
- `language`
- `pii`
- `perplexity`
- `toxicity`
- `report`
- `statistics`
- `tokenizer_stats`

Current filters are heuristic unless an optional dependency is explicitly used.

## Tracking

`arclm.tracking` provides local experiment tracking:

- `create_experiment`
- `ExperimentTracker`
- `list_experiments`

Optional MLflow and W&B integration are used only when available.

## Training Extension Classes

`arclm.training` exports:

- `BaseTrainingPipeline`
- `BaseModelLoader`
- `BaseModelAdapter`
- `UnifiedPipeline`
- `PreTrainedModelLoader`
- `ModelAdapter`
- `StoppingCriteria`

`UnifiedPipeline` exists, but the most complete current high-level API is
`train_model()` for native ArcLM workflows and `train_sft()` for Hugging Face
SFT.

## Repository Example Scripts

Maintained public examples live in `examples/` as numbered scripts:

- `01_quickstart.py`
- `02_tokenization.py`
- `03_data_processing.py`
- `04_pretraining.py`
- `05_continue_training.py`
- `06_finetuning.py`
- `07_native_sft.py`
- `08_huggingface_sft.py`
- `09_lora_sft.py`
- `10_preprocess_pipeline.py`
- `11_inference.py`
- `12_smart_loader.py`
- `13_diagnostics.py`
- `14_custom_trainer.py`
- `15_custom_hf_sft_loop.py`

Advanced root-level scripts are practical workflows:

- `train.py`: full Qwen medical SFT workflow using `FreedomIntelligence/medical-o1-reasoning-SFT`, `Qwen/Qwen3-0.6B`, optional LoRA, gradient accumulation, token-length analysis, and periodic Hugging Face checkpoint folders.
- `test.py`: matching inference script that loads a LoRA adapter, a full Hugging Face model directory, or the latest checkpoint directory.

The shortest public API for Hugging Face SFT is `train_sft()`. The root
`train.py` script uses advanced helpers from `arclm.sft` to expose a more
complete application workflow with periodic checkpoint folders.

## Logic Helpers

`arclm.logics` contains educational propositional logic classes:

- `Symbol`
- `Sentence`
- `Not`
- `And`
- `Or`
- `Implication`
- `Biconditional`
- `model_check`

They are unrelated to language-model training.

## Current Limitations

- No preference training: DPO/RLHF/PPO/reward modeling are not implemented.
- No native ArcLM LoRA layers.
- Native `ArcLM.forward()` does not accept `attention_mask`.
- Hosted provider clients are not implemented.
- Python APIs are the canonical workflows.
- Hugging Face adaptation into native ArcLM is best-effort, based on matching tensor names and shapes.
