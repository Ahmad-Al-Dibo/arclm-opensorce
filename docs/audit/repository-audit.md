# Repository Audit

Audit date: 2026-07-28. Version found in `arclm/_version.py`: `0.6.1`.

## Current Architecture

| Area | Modules | Current purpose | Public/internal boundary |
| --- | --- | --- | --- |
| Package interface | `arclm/__init__.py`, `arclm/api.py`, `arclm/__main__.py` | Top-level exports, simple helpers, module execution. | `arclm.__all__` is public. `api.py` is older/minimal. |
| Configuration | `arclm/config.py`, `arclm/config_loader.py` | Mutable config object plus JSON/YAML loading/saving. | Public. Validation is stronger in `create_config` than `Config`. |
| Data loading | `arclm/data_processor.py`, `arclm/data.py`, `arclm/dataset.py`, `arclm/instruction_dataset.py` | In-memory record loading, native token preparation, PyTorch datasets. | Public. `DataProcessor` is the clearest data-first API. |
| Preprocessing | `arclm/preprocess/*` | JSONL cleaning, filtering, dedup, reports. | `PreprocessConfig` and `PreprocessPipeline` are public; helper modules are lower-level. |
| Tokenization | `arclm/tokenizer.py`, `arclm/tokenizers/__init__.py` | Word tokenizer, SentencePiece tokenizer, factory. | Public. |
| Native model | `arclm/model.py` | Compact GPT-style causal LM. | `ArcLM` public, `MiniGPT` legacy alias. |
| Training | `arclm/trainer.py`, `arclm/pipeline.py`, `arclm/training/*`, `arclm/sft.py` | Native training loop, high-level `train_model`, older unified pipeline, HF SFT. | Public, with mixed maturity. |
| Loading/inference | `arclm/inference.py`, `arclm/external_inference.py`, `arclm/loaders/*` | Native loading/generation, external source inspection/loading, save/export. | Public but external paths are experimental. |
| Evaluation/diagnostics | `arclm/diagnostics.py`, `arclm/utils.py` | Perplexity/metrics/top-k/concept/long-context diagnostics. | Public with limited test coverage. |
| Web interface | `arclm/simple_interface.py`, templates | Optional Flask inference UI. | Public lazy helpers at top level. |
| Logic utilities | `arclm/logics/*` | Propositional logic helpers. | Exported publicly but unrelated to the new framework scope. |
| Examples/tests/docs | `examples/*`, `tests/*`, `docs/versions/0.5.0/*` | Runnable examples, unit tests, old docs. | Examples partly current; old docs are versioned archival content. |

## Data Flow

1. `DataProcessor.load` reads records into `ProcessedDataset`.
2. `ProcessedDataset.clean/filter/transform/tokenize/split` prepares rows in memory.
3. Native training uses `prepare_data(config)`, which reads text tokens, builds or reuses a tokenizer, encodes train/validation tokens, and creates PyTorch dataloaders.
4. `train_model` builds or adapts a native model, trains with `Trainer`, and saves a checkpoint with tokenizer metadata.
5. `load_model` restores native checkpoints for `LoadedModel.predict`.
6. `load_any_model` inspects a path/model ID and loads native, Hugging Face full model, LoRA adapter, or state dict paths when possible.

## Model Loading

- Native `.pth/.pt/.ckpt` ArcLM checkpoints are loaded by `load_model` and `ArcLMCheckpointLoader`.
- Hugging Face sources are detected by path metadata or owner/model ID and loaded with `AutoTokenizer` plus `AutoModelForCausalLM`.
- LoRA adapters require PEFT and a base model.
- Raw state dicts can be loaded/adapted only when enough config/tokenizer metadata exists for generation.

## Training Configuration

Native training is configured with `Config` or keyword overrides to `train_model`. Important fields are `embed_dim`, `block_size`, `num_blocks`, `batch_size`, `num_epochs`, `learning_rate`, `tokenizer_type`, `max_vocab`, `validation_split`, and `device`.

`train_sft` has its own Hugging Face SFT options such as `assistant_only_loss`, `use_lora`, `max_length`, `dtype`, `device_map`, and LoRA fields.

## Evaluation And Inference

Evaluation is mostly native: `calculate_metrics`, `calculate_perplexity`, `predict_top_k`, and report exporters. Native inference uses `load_model` and `LoadedModel.predict`. External inference uses `load_any_model` and `ExternalLoadedModel.predict/chat/batch_predict`.

## Problems And Gaps

- README focused on the simple web interface and referenced release `0.6.0`, while code is `0.6.1`.
- `examples/README.md` said examples target `0.5.0`.
- No buildable documentation site existed.
- Support claims were implicit and too broad around Hugging Face compatibility.
- External model family support was not separated into official/experimental/compatible/not supported.
- CLI exposes `--num-heads`, but `ArcLM` does not use `num_heads`.
- CLI training path still uses older `UnifiedPipeline` rather than the better documented `train_model`.
- Several public APIs lack complete parameter/return/raises docs.
- `PreprocessConfig` and `PreprocessPipeline` were public but had no class docstrings.
- `DataProcessor` is in-memory and not suitable for large datasets without care.
- Formal dataset schema validation is missing.
- Error handling mixes exceptions and direct `print` warnings.
- Top-level `logics` exports make the package scope unclear.
- `checkpoint_is_compatible_for_tuining` contains a typo but must remain for compatibility.
- `MANIFEST.in` references files such as `CHANGELOG.md`, `CONTRIBUTING.md`, and `requirements-dev.txt`; this update adds docs equivalents but release packaging should still be checked.
- Tests could not run in the provided default environment because Python is `3.14.6`, outside the declared support range, and installed Torch lacks `torch.utils`.

