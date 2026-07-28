# Public API Inventory

This inventory is based on `arclm.__all__` and public subpackage `__all__` files.

## Top-Level Public API

| Category | Public names | Stability | Keep public? | Docs status |
| --- | --- | --- | --- | --- |
| Version/runtime | `get_version`, `get_device`, `list_available_models`, `report_environment`, `load_training_checkpoint`, `format_duration` | Stable-ish | Yes | Documented here; docstrings partial |
| Configuration | `Config`, `create_config` | Stable-ish | Yes | Documented |
| Data | `DataBundle`, `DataProcessor`, `ProcessedDataset`, `TextDataset`, `create_dataloader`, `load_tokens`, `prepare_data`, `read_tokens`, `split_train_val`, `InstructionDataset`, `create_instruction_dataloader` | Stable-ish | Yes | Documented |
| Tokenization | `Tokenizer`, `SentencePieceTokenizer`, `TokenizerFactory`, `create_tokenizer`, `get_tokenizer_from_config` | Stable-ish | Yes | Documented |
| Native model | `ArcLM`, `MiniGPT`, `build_model`, `build_trainer`, `Trainer`, `train_model`, `TrainingResult` | Stable-ish/legacy | Yes, keep `MiniGPT` as alias | Documented |
| Native inference | `DEFAULT_MODEL_PATH`, `LoadedModel`, `load_model`, `predict`, `Generator` | Stable-ish | Yes | Documented |
| SFT/external | `train_sft`, `SFTTrainingResult`, `ExternalModelConfig`, `ExternalLoadedModel`, `GenerationConfig`, `ModelSaveConfig`, `ModelSourceInfo`, `load_any_model`, `load_external_for_inference`, `predict_external`, `fine_tune_external_model`, `train_native_model`, `fine_tune_native_model`, `continue_native_training`, `save_loaded_model` | Experimental | Yes | Documented |
| Loaders | `LoadedCheckpoint`, `LoadPlan`, `AdaptedModelBundle`, `ModelInspector`, `SmartLoader`, `register_model_inspector`, `load_external_model`, `adapt_for_training`, `validate_tokenizer_compatibility` | Experimental | Yes | Documented |
| Diagnostics | `calculate_metrics`, `calculate_perplexity`, `MetricsReport`, `TopKPrediction`, concept/long-context helpers, exporters | Experimental/stable-ish | Yes | Documented |
| Regularization | `L1Regularization`, `L2Regularization`, `EarlyStopping`, `LearningRateScheduler`, `GeneralizationMonitor`, `MixupAugmentation`, `LabelSmoothing` | Experimental | Maybe | Needs deeper docs/tests |
| Training extension classes | `UnifiedPipeline`, `PreTrainedModelLoader`, `ModelAdapter`, `StoppingCriteria`, `BaseTrainingPipeline`, `BaseModelLoader`, `BaseModelAdapter` | Experimental/legacy | Keep for compatibility | Documented at high level |
| Supported models | `ModelCapability`, `SUPPORTED_MODELS`, `OFFICIAL`, `EXPERIMENTAL`, `COMPATIBLE_UNTESTED`, `NOT_SUPPORTED`, `get_supported_models`, `get_model_capability`, `is_model_officially_supported` | Stable-ish | Yes | New docs |
| Web | `create_simple_interface_app`, `run_simple_interface` | Experimental | Yes | README and CLI docs |
| Logic utilities | `Sentence`, `Symbol`, `Not`, `And`, `Or`, `Implication`, `Biconditional`, `model_check` | Legacy/auxiliary | Keep for compatibility, consider moving | Marked as scope issue |

## Important Parameters And Returns

- `train_model(mode, data, output, checkpoint=None, config=None, tokenizer=None, **config_overrides) -> TrainingResult`
- `train_sft(model, dataset, output_dir, backend="huggingface", assistant_only_loss=True, use_lora=False, ...) -> SFTTrainingResult`
- `load_any_model(source, **kwargs) -> ExternalLoadedModel`
- `load_model(model_path, device="auto", prefer_best=True) -> LoadedModel`
- `DataProcessor.load(path, format=None, loader=None) -> ProcessedDataset`
- `ProcessedDataset.transform(format="pretraining", mapping=None, template=None, text_fields=None, tokenizer=None) -> ProcessedDataset`
- `Tokenizer(max_vocab=50000, default_token="<UNK>", user_defined_symbols=None)`
- `SentencePieceTokenizer(max_vocab=50000, model_type="bpe", character_coverage=1.0, ...)`
- `PreprocessPipeline(config).run(input_path, output_path, report_dir=None) -> dict`

## CLI Commands

- `arclm train --data PATH --output PATH`
- `arclm eval --model PATH --data PATH`
- `arclm generate --model PATH --prompt TEXT`
- `python -m arclm --run simple-interface`
- `python -m arclm.preprocess.cli INPUT --output OUTPUT`

