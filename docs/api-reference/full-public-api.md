# Full Public API

This page is generated from the public names exported by `arclm.__all__` plus public methods found on exported classes. It is intended as a broad map of the framework surface; detailed usage examples live in the focused API pages.

## Public Exports

| API | Kind | Source | Signature or fields | Summary |
| --- | --- | --- | --- | --- |
| `ArcLM` | class | `arclm.model.ArcLM` | `class` | Compact GPT-style causal language model. |
| `MiniGPT` | class | `arclm.model.MiniGPT` | `class` | Deprecated backward-compatible alias for :class:`ArcLM`. |
| `Lab` | class | `arclm.vnext.lab.Lab` | `fields: runtime, decisions, last_dataset, last_model, last_trainer` | High-level educational ArcLM interface. |
| `ModelBundle` | class | `arclm.models.ModelBundle` | `fields: model, tokenizer, config, device, precision, capability_report, source_info, source, backend` | Loaded model bundle returned by :func:`load_model`. |
| `Model` | class | `arclm.vnext.model.Model` | `fields: model, config, spec, runtime, tokenizer, artifact, base_model_id` | ArcLM-owned model wrapper for the first vNext vertical slice. |
| `Dataset` | class | `arclm.vnext.dataset.Dataset` | `fields: records, source, format, inspection, prepared` | ArcLM-owned dataset handle for high-level and Lab workflows. |
| `ModelRegistry` | class | `arclm.vnext.registry.ModelRegistry` | `fields: _specs` | Authoritative vNext model registry. |
| `ModelSpec` | class | `arclm.vnext.registry.ModelSpec` | `fields: architecture_id, display_name, tasks, capabilities, architecture, task, configuration, weight_mapping, weight_mapper, tokenizer_requirements` | ArcLM-owned description of an architecture and its capabilities. |
| `Runtime` | class | `arclm.runtime.device.Runtime` | `fields: backend, device, precision, available_devices, warnings` | Resolved ArcLM runtime plan for local execution. |
| `ModelSupportReport` | class | `arclm.models.ModelSupportReport` | `fields: source, task, detected_architecture, model_type, causal_lm_compatible, tokenizer_available, training_support, inference_support, required_optional_dependencies, device_compatibility` | Runtime model-support inspection result. |
| `inspect_model_support` | function | `arclm.models.inspect_model_support` | `(source: str \| Path, *, task: str='causal-lm', device: str='auto', precision: str='auto', trust_remote_code: bool=False, tokenizer_path: str \| Path \| None=None) -> ModelSupportReport` | Inspect whether a model source can be used by ArcLM for causal LM workflows. |
| `load_model_bundle` | function | `arclm.models.load_model` | `(source: str \| Path, *, task: str='causal-lm', device: str='auto', precision: str='auto', trust_remote_code: bool=False, tokenizer_path: str \| Path \| None=None, local_files_only: bool=False) -> ModelBundle` | Load a native ArcLM checkpoint or Hugging Face causal LM source. |
| `Config` | class | `arclm.config.Config` | `class` | Central configuration for ArcLM. |
| `ArcLMConfig` | class | `arclm.config.ArcLMConfig` | `fields: schema_version, run, data, validation, quality, preprocessing, split, tokenization, model, training` | Typed ArcLM workflow configuration schema. |
| `CacheConfig` | class | `arclm.config.CacheConfig` | `fields: enabled, dir, read_only` | Cache behavior for deterministic workflow steps. |
| `ConfigMigrationReport` | class | `arclm.config.ConfigMigrationReport` | `fields: source_schema_version, target_schema_version, config, fields_renamed, fields_removed, defaults_inserted, values_transformed, warnings, manual_actions_required` | Configuration migration result. |
| `DataConfig` | class | `arclm.config.DataConfig` | `fields: path, format, schema, streaming, strict, malformed` | Dataset source configuration. |
| `ModelConfig` | class | `arclm.config.ModelConfig` | `fields: name, task, revision, device, precision, trust_remote_code, local_files_only` | Model loading and support-validation configuration. |
| `PreprocessingConfig` | class | `arclm.config.PreprocessingConfig` | `fields: deduplicate, deduplicate_fields, normalize_duplicates` | Preprocessing options consumed by ArcLM workflow helpers. |
| `QualityConfig` | class | `arclm.config.QualityConfig` | `fields: enabled, checks, include_samples, redact_samples` | Dataset quality-analysis options. |
| `SecurityConfig` | class | `arclm.config.SecurityConfig` | `fields: loading_policy, allow_env_expansion, redact_secrets` | Security-sensitive workflow defaults. |
| `SplitConfig` | class | `arclm.config.SplitConfig` | `fields: train, validation, test, strategy, key, group_key, split_field, seed` | Deterministic dataset splitting options. |
| `WorkflowEvaluationConfig` | class | `arclm.config.EvaluationConfig` | `fields: enabled, max_batches, metrics` | Evaluation-stage configuration. |
| `WorkflowInferenceConfig` | class | `arclm.config.InferenceConfig` | `fields: max_new_tokens, temperature, batch_size` | Inference-stage configuration. |
| `WorkflowRunConfig` | class | `arclm.config.RunConfig` | `fields: name, output_dir, seed` | Run-directory and reproducibility configuration. |
| `create_config` | function | `arclm.config.create_config` | `(**kwargs) -> Config` | Create a Config object with sensible defaults that can be overridden. |
| `load_arclm_config` | function | `arclm.config.load_arclm_config` | `(source: str \| Path \| Mapping[str, Any], *, permissive: bool=False, allow_env: bool=False) -> ArcLMConfig` | Load and validate an ArcLM configuration from JSON, TOML, or a mapping. |
| `migrate_config` | function | `arclm.config.migrate_config` | `(source: str \| Path \| Mapping[str, Any], *, target_version: str=CONFIG_SCHEMA_VERSION, output: str \| Path \| None=None, permissive: bool=False, base_dir: str \| Path \| None=None) -> ConfigMigrationReport` | Migrate an older ArcLM configuration shape to schema version ``1``. |
| `validate_arclm_config` | function | `arclm.config.validate_arclm_config` | `(source: str \| Path \| Mapping[str, Any], *, permissive: bool=False, allow_env: bool=False) -> ArcLMConfig` | Validate and return a typed ArcLM configuration. |
| `CheckpointInspectionReport` | class | `arclm.checkpoints.CheckpointInspectionReport` | `fields: path, exists, load_attempted, loading_policy, format_version, trusted_pickle_required, is_directory_checkpoint, is_native_arclm, has_model_state, model_weight_format` | Structured checkpoint reliability report. |
| `inspect_checkpoint` | function | `arclm.checkpoints.inspect_checkpoint` | `(path: str \| Path, *, trusted: bool=False, map_location: str='cpu', trust: str \| LoadingPolicy \| None=None) -> CheckpointInspectionReport` | Inspect an ArcLM checkpoint without unsafe loading by default. |
| `load_trusted_checkpoint` | function | `arclm.checkpoints.load_trusted_checkpoint` | `(path: str \| Path, *, map_location: str='cpu') -> Dict[str, Any]` | Load a trusted checkpoint after structured inspection. |
| `verify_checkpoint` | function | `arclm.checkpoints.verify_checkpoint` | `(path: str \| Path, *, trust: str \| LoadingPolicy='safe') -> CheckpointInspectionReport` | Inspect a checkpoint and raise when it is incomplete or unsafe. |
| `write_checkpoint_manifest` | function | `arclm.checkpoints.write_checkpoint_manifest` | `(checkpoint_dir: str \| Path, *, model_config: Optional[Mapping[str, Any]]=None, source_model: Optional[str]=None, source_revision: Optional[str]=None, security_classification: str='trusted_local') -> Path` | Create or refresh a versioned ArcLM checkpoint manifest and hashes. |
| `DoctorReport` | class | `arclm.doctor.DoctorReport` | `fields: checks, arclm_version, report_type, schema_version` | Structured doctor command output. |
| `run_doctor` | function | `arclm.doctor.run_doctor` | `(*, config: str \| Path \| None=None, checkpoint: str \| Path \| None=None, run_dir: str \| Path='runs', cache_dir: str \| Path='.arclm/cache') -> DoctorReport` | Run local diagnostics without downloading models. |
| `LoadingPolicy` | class | `arclm.security.LoadingPolicy` | `fields: mode, allow_pickle, allow_remote_code, verify_hashes` | Explicit checkpoint/model loading trust policy. |
| `DeviceConfig` | class | `arclm.resources.DeviceConfig` | `fields: device, precision, allow_fallback` | Explicit CPU/CUDA device and precision request. |
| `DeviceSelection` | class | `arclm.resources.DeviceSelection` | `fields: requested_device, selected_device, requested_precision, selected_precision, fallback_used, warnings` | Validated runtime device and precision selection. |
| `ResourceLimits` | class | `arclm.resources.ResourceLimits` | `fields: max_records, max_record_chars, max_bytes` | Soft resource limits used by data workflows. |
| `resource_info` | function | `arclm.resources.resource_info` | `() -> dict[str, Any]` | Return basic local resource information. |
| `Stability` | class | `arclm.stability.Stability` | `fields: STABLE, PROVISIONAL, EXPERIMENTAL, DEPRECATED, INTERNAL` | Release-candidate stability labels for public interfaces. |
| `api_manifest` | function | `arclm.stability.api_manifest` | `() -> list[dict[str, str \| None]]` | Return the machine-readable API stability manifest. |
| `cli_manifest` | function | `arclm.stability.cli_manifest` | `() -> dict[str, str]` | Return command stability labels. |
| `stable_api_paths` | function | `arclm.stability.stable_api_paths` | `() -> list[str]` | Return stable public API paths used by snapshot tests. |
| `CacheStats` | class | `arclm.cache.CacheStats` | `fields: path, entries, bytes, keys, corrupted, report_type, schema_version, arclm_version` | Cache directory statistics. |
| `clear_cache` | function | `arclm.cache.clear_cache` | `(cache_dir: str \| Path, *, key: Optional[str]=None) -> CacheStats` | Clear a cache directory or one key, then return updated stats. |
| `inspect_cache` | function | `arclm.cache.inspect_cache` | `(cache_dir: str \| Path) -> CacheStats` | Return cache statistics without loading cached tensors. |
| `Tokenizer` | class | `arclm.tokenizer.Tokenizer` | `class` | Tokenizer for converting text to token indices and back |
| `SentencePieceTokenizer` | class | `arclm.tokenizer.SentencePieceTokenizer` | `class` | SentencePiece BPE tokenizer for subword language-model training. |
| `TokenizerFactory` | class | `arclm.tokenizer.TokenizerFactory` | `class` | Single selection point for tokenizer implementations. |
| `create_tokenizer` | function | `arclm.tokenizer.create_tokenizer` | `(tokenizer_type='word', **kwargs)` | Factory function to create tokenizer instances. |
| `get_tokenizer_from_config` | function | `arclm.tokenizer.get_tokenizer_from_config` | `(config)` | Convenience function to create tokenizer from Config object. |
| `TextDataset` | class | `arclm.dataset.TextDataset` | `class` | Dataset for language modeling with sliding window |
| `create_dataloader` | function | `arclm.dataset.create_dataloader` | `(encoded_data, block_size, batch_size, shuffle=True)` | Create DataLoader from encoded data |
| `DataBundle` | class | `arclm.data.DataBundle` | `fields: tokens, train_tokens, val_tokens, tokenizer, train_loader, val_loader, train_encoded, val_encoded` | Prepared tokens, tokenizer, and loaders used by training. |
| `DatasetSource` | class | `arclm.data_sources.DatasetSource` | `class` | Iterable dataset source for records or streaming files. |
| `DataPipeline` | class | `arclm.data_pipeline.DataPipeline` | `class` | Deterministic, composable pipeline for in-memory dataset records. |
| `DataPipelineReport` | class | `arclm.data_pipeline.DataPipelineReport` | `fields: input_count, output_count, removed_record_count, duration_seconds, operations, validation_failures, warnings, configuration, reproducibility` | Structured report for a :class:`DataPipeline` run. |
| `DataQualityReport` | class | `arclm.data_quality.DataQualityReport` | `fields: total_records, metrics, issues, samples, warnings, errors, report_type, schema_version, arclm_version, created_at` | Privacy-aware data-quality report. |
| `DatasetValidationReport` | class | `arclm.schemas.DatasetValidationReport` | `fields: schema, strict, total_records, valid_records, invalid_records, errors, warnings, invalid_record_indexes, detected_fields, empty_content` | Structured result for dataset validation. |
| `DuplicateReport` | class | `arclm.data_quality.DuplicateReport` | `fields: total_records, duplicate_records, groups, fields, normalize, created_at, report_type, schema_version, arclm_version` | Result returned by :func:`find_duplicates`. |
| `ShardResult` | class | `arclm.data_quality.ShardResult` | `fields: shards, report` |  |
| `SplitResult` | class | `arclm.data_quality.SplitResult` | `fields: splits, report` |  |
| `ValidationIssue` | class | `arclm.schemas.ValidationIssue` | `fields: index, schema, field, category, message, level` | A validation error or warning for one record. |
| `TextRecord` | class | `arclm.schemas.TextRecord` | `fields: text, schema_name, allowed_fields, required_fields` | A pretraining text record. |
| `PromptCompletionRecord` | class | `arclm.schemas.PromptCompletionRecord` | `fields: prompt, completion, schema_name, allowed_fields, required_fields` | A prompt-completion fine-tuning record. |
| `InstructionRecord` | class | `arclm.schemas.InstructionRecord` | `fields: instruction, input, output, schema_name, allowed_fields, required_fields` | An instruction/input/output supervised fine-tuning record. |
| `ConversationRecord` | class | `arclm.schemas.ConversationRecord` | `fields: messages, schema_name, allowed_fields, required_fields` | A chat conversation record with role/content messages. |
| `analyze_dataset` | function | `arclm.data_quality.analyze_dataset` | `(dataset: Iterable[Mapping[str, Any]], *, schema: Optional[str]=None, checks: Optional[Sequence[str]]=None, text_field: str='text', include_samples: bool=False, redact_samples: bool=True, max_sample_chars: int=120, tokenizer: Optional[Any]=None) -> DataQualityReport` | Analyze dataset quality without exposing full records by default. |
| `check_leakage` | function | `arclm.data_quality.check_leakage` | `(train_records: Iterable[Mapping[str, Any]], test_records: Iterable[Mapping[str, Any]], *, fields: Optional[Sequence[str]]=None, normalize: bool=True) -> DuplicateReport` | Detect exact overlap between train and test records. |
| `find_duplicates` | function | `arclm.data_quality.find_duplicates` | `(dataset: Iterable[Mapping[str, Any]], *, fields: Optional[Sequence[str]]=None, normalize: bool=False) -> DuplicateReport` | Find exact duplicate groups by whole record or selected fields. |
| `find_near_duplicates` | function | `arclm.data_quality.find_near_duplicates` | `(dataset: Iterable[Mapping[str, Any]], *, field: str='text', threshold: float=0.9, normalize: bool=True) -> DuplicateReport` | Find approximate near-duplicates with token-set Jaccard similarity. |
| `DataProcessor` | class | `arclm.data_processor.DataProcessor` | `class` | Load JSON, JSONL, CSV, TXT, or custom datasets for preprocessing. |
| `ProcessedDataset` | class | `arclm.data_processor.ProcessedDataset` | `fields: samples, source` | Small in-memory dataset wrapper for prompt formatting workflows. |
| `load_tokens` | function | `arclm.data.load_tokens` | `(config)` | Load the configured dataset, optionally mixing repeated domain data first. |
| `open_dataset` | function | `arclm.data_sources.open_dataset` | `(source: str \| Path \| Iterable[Mapping[str, Any]], *, format: Optional[str]=None, streaming: bool=True, encoding: str='utf-8', blank_lines: str='skip', malformed: str='raise') -> DatasetSource` | Open a dataset as an iterable :class:`DatasetSource`. |
| `prepare_data` | function | `arclm.data.prepare_data` | `(config: Config, existing_tokenizer=None) -> DataBundle` | Load tokens, create train/validation splits, build tokenizer, and loaders. |
| `read_tokens` | function | `arclm.data.read_tokens` | `(path, limit)` | Read lowercase whitespace tokens from a text file, keeping newlines. |
| `shard_dataset` | function | `arclm.data_quality.shard_dataset` | `(dataset: Iterable[Mapping[str, Any]], *, num_shards: int, strategy: str='contiguous', key: Optional[str]=None, seed: int=0) -> ShardResult` | Deterministically shard records without duplication or loss. |
| `split_dataset` | function | `arclm.data_quality.split_dataset` | `(dataset: Iterable[Mapping[str, Any]], *, train: float=0.8, validation: float=0.1, test: float=0.1, seed: int=42, strategy: str='hash', key: Optional[str]=None, group_key: Optional[str]=None, date_field: Optional[str]=None, split_field: Optional[str]=None) -> SplitResult` | Create deterministic train/validation/test splits. |
| `split_train_val` | function | `arclm.data.split_train_val` | `(tokens, validation_split, block_size, seed=42)` | Split tokens by shuffled sentences when possible, otherwise by suffix. |
| `validate_record` | function | `arclm.schemas.validate_record` | `(record: Mapping[str, Any], *, schema: str, index: int=0, strict: bool=True, allow_empty: bool=False) -> Tuple[Optional[BaseRecord], List[ValidationIssue], List[ValidationIssue]]` | Validate one record against a named schema. |
| `validate_records` | function | `arclm.schemas.validate_records` | `(records: Iterable[Mapping[str, Any]], *, schema: str, strict: bool=True, allow_empty: bool=False, check_duplicates: bool=False, duplicate_field: Optional[str]=None) -> DatasetValidationReport` | Validate a batch of records and return a structured report. |
| `ArcLMError` | class | `arclm.exceptions.ArcLMError` | `class` | Base class for ArcLM-specific errors. |
| `ArtifactError` | class | `arclm.exceptions.ArtifactError` | `class` | Raised when an ArcLM native artifact cannot be read or written. |
| `ArtifactIntegrityError` | class | `arclm.exceptions.ArtifactIntegrityError` | `class` | Raised when an ArcLM native artifact fails integrity validation. |
| `CheckpointError` | class | `arclm.exceptions.CheckpointError` | `class` | Raised when checkpoint loading or saving fails. |
| `ConfigurationError` | class | `arclm.exceptions.ConfigurationError` | `class` | Raised when configuration values are invalid. |
| `DatasetError` | class | `arclm.exceptions.DatasetError` | `class` | Base class for dataset errors. |
| `DatasetFormatError` | class | `arclm.exceptions.DatasetFormatError` | `class` | Raised when dataset input cannot be parsed into records. |
| `DatasetValidationError` | class | `arclm.exceptions.DatasetValidationError` | `class` | Raised when records do not satisfy a requested schema. |
| `ModelCompatibilityError` | class | `arclm.exceptions.ModelCompatibilityError` | `class` | Raised when a model is not compatible with a requested task. |
| `ModelError` | class | `arclm.exceptions.ModelError` | `class` | Base class for model errors. |
| `ModelLoadError` | class | `arclm.exceptions.ModelLoadError` | `class` | Raised when model loading fails. |
| `OptionalDependencyError` | class | `arclm.exceptions.OptionalDependencyError` | `class` | Raised when an optional dependency is required but missing. |
| `TrainingError` | class | `arclm.exceptions.TrainingError` | `class` | Raised when training cannot be completed. |
| `UnsupportedModelError` | class | `arclm.exceptions.UnsupportedModelError` | `class` | Raised when a model is outside ArcLM's supported workflow. |
| `Trainer` | class | `arclm.vnext.trainer.Trainer` | `class` | Public Trainer that supports new high-level and legacy construction. |
| `LegacyTrainer` | class | `arclm.trainer.Trainer` | `class` | Model trainer with save/load functionality and generalization monitoring |
| `Generator` | class | `arclm.generator.Generator` | `class` | Text generation with trained model |
| `DEFAULT_MODEL_PATH` | constant | `arclm.inference.DEFAULT_MODEL_PATH` | `` |  |
| `LoadedModel` | class | `arclm.inference.LoadedModel` | `fields: model, generator, config, model_path, device` | Loaded model bundle with a minimal prediction method. |
| `GenerationResult` | class | `arclm.inference.GenerationResult` | `fields: outputs, prompt_tokens, generated_tokens, latency_seconds, tokens_per_second, errors, report_type, schema_version` | Structured result returned by batched generation. |
| `load_model` | function | `arclm.inference.load_model` | `(model_path=None, device=None, prefer_best=True)` | Load a trained ArcLM checkpoint for inference. |
| `predict` | function | `arclm.inference.predict` | `(input_text, model_path=None, device=None, reload=False, **generation_options)` | Predict with a cached model using the same defaults as load_model(). |
| `build_model` | function | `arclm.pipeline.build_model` | `(config, vocab_size=None) -> ArcLM` | Build an ArcLM model from config and move it to the configured device. |
| `build_trainer` | function | `arclm.pipeline.build_trainer` | `(model, config, event_logger=None)` | Create the optimizer, criterion, and Trainer for a model. |
| `train_model` | function | `arclm.pipeline.train_model` | `(mode: str, data: str, output: str, checkpoint: Optional[str]=None, config: Optional[Config]=None, tokenizer=None, **config_overrides) -> TrainingResult` | Train an ArcLM model in pretrain, finetune, or continue_training mode. |
| `TrainingResult` | class | `arclm.pipeline.TrainingResult` | `fields: mode, model_path, config, history, vocab_size, tokenizer, checkpoint_source` | Result returned by the high-level train_model API. |
| `TrainingConfig` | class | `arclm.training.engine.TrainingConfig` | `fields: output_dir, task, epochs, batch_size, gradient_accumulation_steps, learning_rate, weight_decay, warmup_ratio, max_grad_norm, evaluation_strategy` | Validated training configuration used by :func:`train`. |
| `TrainingReport` | class | `arclm.training.engine.TrainingReport` | `fields: backend, output_dir, steps, duration_seconds, metrics, checkpoints, warnings, errors, config, report_type` | Structured training result. |
| `train` | function | `arclm.training.engine.train` | `(*, model: Any=None, dataset: Any=None, config: TrainingConfig \| None=None, callbacks: Optional[Iterable[EventHandler]]=None, model_name: Optional[str]=None) -> TrainingReport` | Train through ArcLM's unified facade. |
| `train_sft` | function | `arclm.sft.train_sft` | `(model: str, dataset: str, output_dir: str, backend: str='huggingface', assistant_only_loss: bool=True, use_lora: bool=False, batch_size: int=1, gradient_accumulation_steps: int=1, learning_rate: float=0.0002, num_epochs: int=1, max_length: int=1024, dtype: str='auto', device_map: Optional[str]=None, trust_remote_code: bool=True, enable_thinking: Optional[bool]=False, lora_r: int=8, lora_alpha: int=16, lora_dropout: float=0.05, lora_target_modules: Optional[Sequence[str]]=None, max_steps: Optional[int]=None, seed: int=42, save_tokenizer: bool=True) -> SFTTrainingResult` | Run supervised fine-tuning through an ArcLM backend. |
| `SFTTrainingResult` | class | `arclm.sft.SFTTrainingResult` | `fields: model, dataset, output_dir, backend, use_lora, assistant_only_loss, train_loss_history, steps, metadata_path, adapter_path` | Result returned by ``train_sft``. |
| `COMPATIBLE_UNTESTED` | constant | `arclm.supported_models.COMPATIBLE_UNTESTED` | `` |  |
| `EXPERIMENTAL` | constant | `arclm.supported_models.EXPERIMENTAL` | `` |  |
| `NOT_SUPPORTED` | constant | `arclm.supported_models.NOT_SUPPORTED` | `` |  |
| `OFFICIAL` | constant | `arclm.supported_models.OFFICIAL` | `` |  |
| `SUPPORTED_MODELS` | constant/object | `arclm.supported_models.SUPPORTED_MODELS` | `` |  |
| `ModelCapability` | class | `arclm.supported_models.ModelCapability` | `fields: family, example_models, architecture, status, training, inference, quantization, device_requirements, tokenizer_requirements, attention_implementation` | Capability record for a model family or architecture. |
| `get_model_capability` | function | `arclm.supported_models.get_model_capability` | `(family: str) -> ModelCapability` | Look up model capability metadata by family name. |
| `get_supported_models` | function | `arclm.supported_models.get_supported_models` | `(status: Optional[str]=None) -> List[ModelCapability]` | Return ArcLM's declared model-support records. |
| `is_model_officially_supported` | function | `arclm.supported_models.is_model_officially_supported` | `(family: str) -> bool` | Return whether a model family is officially supported by ArcLM. |
| `ExternalModelConfig` | class | `arclm.external_inference.ExternalModelConfig` | `fields: source, base_model, device, device_map, dtype, trust_remote_code, tokenizer_path, max_memory, load_in_8bit, load_in_4bit` | Configuration for loading native or external causal language models. |
| `ExternalLoadedModel` | class | `arclm.external_inference.ExternalLoadedModel` | `fields: model, tokenizer, config, source_info, device, processor, _native_loaded, _generation_error, _training_method, _lora_merged` | Loaded model wrapper with a unified inference and save API. |
| `GenerationConfig` | class | `arclm.external_inference.GenerationConfig` | `fields: max_new_tokens, temperature, top_p, top_k, do_sample, repetition_penalty, pad_token_id, eos_token_id, stop, return_full_text` | Text generation defaults used by :class:`ExternalLoadedModel`. |
| `ModelSaveConfig` | class | `arclm.external_inference.ModelSaveConfig` | `fields: save_enabled, save_mode, save_layout, save_tokenizer, save_model_config, save_generation_config, save_training_metadata, save_adapter_config, save_processor, save_readme` | Developer-controlled settings for saving or exporting loaded models. |
| `ModelSourceInfo` | class | `arclm.external_inference.ModelSourceInfo` | `fields: source, source_type, tokenizer_files_exist, adapter_config_exists, config_json_exists, model_weights_exist, detected_checkpoint_folders, recommended_loading_strategy, warnings, resolved_source` | Structured inspection result for a model source. |
| `inspect_model_source` | function | `arclm.external_inference.inspect_model_source` | `(source: PathLike, **kwargs: Any) -> ModelSourceInfo` | Inspect a local path or Hugging Face model ID before loading. |
| `load_any_model` | function | `arclm.external_inference.load_any_model` | `(source: PathLike, **kwargs: Any) -> ExternalLoadedModel` | Load any supported ArcLM or Hugging Face causal LM source for inference. |
| `load_external_for_inference` | function | `arclm.external_inference.load_external_for_inference` | `(source: PathLike, **kwargs: Any) -> ExternalLoadedModel` | Backward-friendly alias for :func:`load_any_model`. |
| `predict_external` | function | `arclm.external_inference.predict_external` | `(source: PathLike, prompt: str, **kwargs: Any) -> str` | Load a source and return a single prediction. |
| `fine_tune_external_model` | function | `arclm.external_inference.fine_tune_external_model` | `(model: str, dataset: str, output_dir: str, method: str='sft', backend: str='huggingface', use_lora: bool=True, assistant_only_loss: bool=True, save_config: Optional[ModelSaveConfig]=None, **kwargs: Any) -> SFTTrainingResult` | Fine-tune an external model through supported ArcLM training backends. |
| `train_native_model` | function | `arclm.external_inference.train_native_model` | `(*, data: str, output: str, **kwargs: Any) -> Any` | Train a native ArcLM model by delegating to ``train_model(mode='pretrain')``. |
| `fine_tune_native_model` | function | `arclm.external_inference.fine_tune_native_model` | `(*, data: str, output: str, checkpoint: str, **kwargs: Any) -> Any` | Fine-tune a native ArcLM checkpoint via ``train_model``. |
| `continue_native_training` | function | `arclm.external_inference.continue_native_training` | `(*, data: str, output: str, checkpoint: str, **kwargs: Any) -> Any` | Continue native ArcLM training via ``train_model``. |
| `save_loaded_model` | function | `arclm.external_inference.save_loaded_model` | `(loaded_model: ExternalLoadedModel, output_dir: PathLike, save_config: Optional[ModelSaveConfig]=None) -> Optional[Path]` | Save a loaded model according to :class:`ModelSaveConfig`. |
| `checkpoint_is_compatible_for_continue_training` | function | `arclm.pipeline.checkpoint_is_compatible_for_continue_training` | `(checkpoint, config, vocab_size, tokenizer=None)` | Check whether a checkpoint can be reused for the current run. |
| `checkpoint_is_compatible_for_tuning` | function | `arclm.pipeline.checkpoint_is_compatible_for_tuning` | `(checkpoint, config, vocab_size, tokenizer=None)` | Check whether a checkpoint can be reused for native fine-tuning. |
| `checkpoint_is_compatible_for_tuining` | function | `arclm.pipeline.checkpoint_is_compatible_for_tuining` | `(checkpoint, config, vocab_size, tokenizer=None)` | Deprecated misspelled alias for :func:`checkpoint_is_compatible_for_tuning`. |
| `create_checkpoint_callback` | function | `arclm.pipeline.create_checkpoint_callback` | `(config, tokenizer, vocab_size)` | Create a callback that saves the latest resumable checkpoint. |
| `create_epoch_checkpoint_callback` | function | `arclm.pipeline.create_epoch_checkpoint_callback` | `(config, tokenizer, vocab_size)` | Backward-compatible alias for create_checkpoint_callback. |
| `load_compatible_checkpoint` | function | `arclm.pipeline.load_compatible_checkpoint` | `(trainer, config, vocab_size, tokenizer=None)` | Load an existing checkpoint when it matches the current model setup. |
| `save_training_checkpoint` | function | `arclm.pipeline.save_training_checkpoint` | `(trainer, config, tokenizer, vocab_size)` | Save a trainer checkpoint with tokenizer metadata. |
| `tokenizer_from_checkpoint` | function | `arclm.pipeline.tokenizer_from_checkpoint` | `(checkpoint_or_source)` | Restore an ArcLM tokenizer from a loaded checkpoint or checkpoint path. |
| `UnifiedPipeline` | class | `arclm.training.unified.UnifiedPipeline` | `fields: MODES` | Unified training pipeline supporting multiple modes: |
| `PreTrainedModelLoader` | class | `arclm.training.unified.PreTrainedModelLoader` | `fields: source, target_vocab_size, target_config, device, strict_loading` | Load pre-trained models from multiple sources: |
| `ModelAdapter` | class | `arclm.training.unified.ModelAdapter` | `class` | Adapt weights from external models to ArcLM architecture. |
| `StoppingCriteria` | class | `arclm.training.unified.StoppingCriteria` | `fields: max_steps, early_stopping_patience, early_stopping_min_delta` | Base stopping criteria for training loop control. |
| `BaseTrainingPipeline` | class | `arclm.training.base.BaseTrainingPipeline` | `class` | Base class for developer-customizable training pipelines. |
| `BaseModelLoader` | class | `arclm.training.base.BaseModelLoader` | `class` | Base class for model loaders that return a model and metadata. |
| `BaseModelAdapter` | class | `arclm.training.base.BaseModelAdapter` | `class` | Base class for adapting one model implementation into another. |
| `LoadedCheckpoint` | class | `arclm.loaders.base.LoadedCheckpoint` | `fields: source, source_type, state_dict, config, vocab_size, tokenizer_metadata, vocab, stoi, itos, optimizer_state_dict` | Normalized representation of a model source that ArcLM can inspect. |
| `LoadPlan` | class | `arclm.loaders.smart_loader.LoadPlan` | `fields: source, source_type, files, metadata, model_type, architecture, tokenizer, weight_format, precision, load_as` | Detected and user-selected settings for loading a model source. |
| `AdaptedModelBundle` | class | `arclm.loaders.base.AdaptedModelBundle` | `fields: model, config, checkpoint, missing_keys, unexpected_keys, tokenizer_metadata` | ArcLM model and metadata produced from a normalized checkpoint. |
| `ModelInspector` | class | `arclm.loaders.smart_loader.ModelInspector` | `class` | Base class for source inspectors used by ``SmartLoader``. |
| `SmartLoader` | class | `arclm.loaders.smart_loader.SmartLoader` | `class` | High-level loading entry point with inspection and manual overrides. |
| `register_model_inspector` | function | `arclm.loaders.smart_loader.register_model_inspector` | `(inspector: ModelInspector) -> None` | Register a custom model-source inspector. |
| `load_external_model` | function | `arclm.loaders.registry.load_external_model` | `(source: Any, map_location: str='cpu') -> LoadedCheckpoint` | Load any supported external source into a normalized checkpoint bundle. |
| `adapt_for_training` | function | `arclm.loaders.adapters.adapt_for_training` | `(checkpoint: LoadedCheckpoint, target_config: Optional[Config]=None, tokenizer: Optional[Any]=None, strict: bool=False, require_tokenizer_match: bool=False, **config_overrides) -> AdaptedModelBundle` | Create an ArcLM model from a normalized checkpoint. |
| `validate_tokenizer_compatibility` | function | `arclm.loaders.adapters.validate_tokenizer_compatibility` | `(checkpoint: LoadedCheckpoint, tokenizer: Optional[Any]=None, config: Optional[Config]=None) -> None` | Raise if a tokenizer cannot safely address the checkpoint vocabulary. |
| `build_training_diagnostics_report` | function | `arclm.diagnostics.build_training_diagnostics_report` | `(model, data, config)` | Build tokenizer, top-k, and concept benchmark diagnostics for a run. |
| `calculate_metrics` | function | `arclm.diagnostics.calculate_metrics` | `(model, val_loader, config, device='cpu')` | Calculate comprehensive evaluation metrics. |
| `calculate_perplexity` | function | `arclm.diagnostics.calculate_perplexity` | `(loss)` | Calculate perplexity from cross-entropy loss. |
| `export_metrics_to_json` | function | `arclm.diagnostics.export_metrics_to_json` | `(metrics_report, filepath)` | Export metrics report to JSON file. |
| `export_metrics_to_markdown` | function | `arclm.diagnostics.export_metrics_to_markdown` | `(metrics_report, filepath)` | Export metrics report to Markdown file. |
| `ConceptBenchmarkCase` | class | `arclm.diagnostics.ConceptBenchmarkCase` | `fields: prompt, expected_concepts` | Expected concept relationships for a prompt. |
| `ConceptBenchmarkResult` | class | `arclm.diagnostics.ConceptBenchmarkResult` | `fields: prompt, expected_concepts, predicted_tokens, matched_concepts, score` | Concept benchmark score for one prompt. |
| `DEFAULT_CONCEPT_BENCHMARKS` | constant | `arclm.diagnostics.DEFAULT_CONCEPT_BENCHMARKS` | `` |  |
| `LongContextResult` | class | `arclm.diagnostics.LongContextResult` | `fields: block_size, validation_loss, concept_score, generated_sample` | Summary for one long-context training/evaluation run. |
| `MetricsReport` | class | `arclm.diagnostics.MetricsReport` | `fields: perplexity, avg_loss, total_tokens, correct_predictions, accuracy, timestamp` | Comprehensive metrics report for model evaluation. |
| `TopKPrediction` | class | `arclm.diagnostics.TopKPrediction` | `fields: rank, token, probability` | One next-token prediction. |
| `format_concept_benchmark_report` | function | `arclm.diagnostics.format_concept_benchmark_report` | `(report)` | Create a readable concept benchmark report. |
| `format_long_context_results` | function | `arclm.diagnostics.format_long_context_results` | `(results)` | Create a readable long-context comparison report. |
| `format_tokenizer_coverage_report` | function | `arclm.diagnostics.format_tokenizer_coverage_report` | `(coverage)` | Create a readable tokenizer coverage report. |
| `format_top_k_predictions` | function | `arclm.diagnostics.format_top_k_predictions` | `(prompt, predictions)` | Create a readable diagnostic report for one prompt. |
| `predict_top_k` | function | `arclm.diagnostics.predict_top_k` | `(model, stoi, itos, block_size, device, prompt, k=5, tokenizer=None)` | Return the top-k next-token probabilities for a prompt. |
| `run_long_context_evaluation` | function | `arclm.diagnostics.run_long_context_evaluation` | `(base_config, block_sizes=(32, 64, 128), benchmark_cases=None, sample_prompt='Programming is')` | Train and compare multiple block sizes by validation loss and concepts. |
| `score_concept_relationships` | function | `arclm.diagnostics.score_concept_relationships` | `(model, stoi, itos, block_size, device, benchmark_cases=None, k=10, tokenizer=None)` | Score expected concepts against top-k next-token predictions. |
| `format_duration` | function | `arclm.utils.format_duration` | `(total_seconds)` | Format a duration in seconds as hours, minutes, and seconds. |
| `configure_logging` | function | `arclm.logging.configure_logging` | `(level: str='normal') -> _logging.Logger` | Configure ArcLM package logging for CLI/application entry points. |
| `get_logger` | function | `arclm.logging.get_logger` | `(name: str='arclm') -> _logging.Logger` | Return an ArcLM package logger without configuring global logging. |
| `FingerprintReport` | class | `arclm.reproducibility.FingerprintReport` | `fields: value, algorithm, schema_version, arclm_version, mode, reproducible, warnings` | Structured fingerprint with generation metadata. |
| `fingerprint` | function | `arclm.reproducibility.fingerprint` | `(value: Any, *, mode: str='content', max_records: Optional[int]=None) -> FingerprintReport` | Create a stable fingerprint for serializable values and ArcLM objects. |
| `Run` | class | `arclm.runs.Run` | `class` | Local run directory with configs, reports, metrics, logs, and artifacts. |
| `RunMetadata` | class | `arclm.runs.RunMetadata` | `fields: run_id, name, path, status, started_at, ended_at, arclm_version, python_version, platform, dependencies` | Serializable run metadata. |
| `inspect_run` | function | `arclm.runs.inspect_run` | `(path: str \| Path) -> dict[str, Any]` | Read one run metadata file. |
| `list_runs` | function | `arclm.runs.list_runs` | `(output_dir: str \| Path='runs') -> list[dict[str, Any]]` | List local run metadata files. |
| `TokenizationConfig` | class | `arclm.tokenization.TokenizationConfig` | `fields: tokenizer, schema, text_field, max_length, truncation, padding, add_eos, add_bos, create_labels, prompt_masking` | Serializable tokenization configuration. |
| `TokenizedDataset` | class | `arclm.tokenization.TokenizedDataset` | `fields: records, config, cache_key, cache_hit, tokenizer_identity, dataset_fingerprint, created_at, report_type, schema_version, arclm_version` | Tokenized dataset plus cache metadata. |
| `tokenize_dataset` | function | `arclm.tokenization.tokenize_dataset` | `(dataset: Iterable[Mapping[str, Any]], *, tokenizer: str \| Any, schema: str='text', text_field: str='text', max_length: Optional[int]=None, truncation: bool=True, padding: bool \| str=False, batch_size: int=32, cache_dir: Optional[str]=None, read_only_cache: bool=False, add_eos: bool=False, add_bos: bool=False, create_labels: bool=True, prompt_masking: bool=False, tokenizer_revision: Optional[str]=None) -> TokenizedDataset` | Format and tokenize records for causal language-model workflows. |
| `WorkflowResult` | class | `arclm.workflow.WorkflowResult` | `fields: status, run_dir, stages, dry_run, warnings, errors, report_type, schema_version, arclm_version, created_at` |  |
| `WorkflowStageResult` | class | `arclm.workflow.WorkflowStageResult` | `fields: name, status, duration_seconds, report_path, error` |  |
| `run_workflow` | function | `arclm.workflow.run_workflow` | `(config: str \| Path \| dict[str, Any], *, dry_run: bool=False, stages: Optional[Sequence[str]]=None, resume: bool=False) -> WorkflowResult` | Run a local ArcLM workflow from JSON/TOML-like configuration. |
| `normalize_device` | function | `arclm.config_validation.normalize_device` | `(device: Optional[str]) -> str` | Normalize and validate a device string. |
| `normalize_precision` | function | `arclm.config_validation.normalize_precision` | `(precision: Optional[str]) -> str` | Normalize and validate a precision string. |
| `validate_training_config` | function | `arclm.config_validation.validate_training_config` | `(config: Any) -> Any` | Validate important native training configuration fields in place. |
| `L1Regularization` | class | `arclm.regularization.L1Regularization` | `class` | L1 Regularization - Sparsity inducing |
| `L2Regularization` | class | `arclm.regularization.L2Regularization` | `class` | L2 Regularization (Weight Decay) - Smoothness inducing |
| `EarlyStopping` | class | `arclm.regularization.EarlyStopping` | `class` | Early stopping to prevent overfitting |
| `LearningRateScheduler` | class | `arclm.regularization.LearningRateScheduler` | `class` | Learning rate scheduler for better generalization |
| `GeneralizationMonitor` | class | `arclm.regularization.GeneralizationMonitor` | `class` | Monitor model generalization during training |
| `MixupAugmentation` | class | `arclm.regularization.MixupAugmentation` | `class` | Mixup data augmentation for better generalization |
| `LabelSmoothing` | class | `arclm.regularization.LabelSmoothing` | `class` | Label smoothing regularization |
| `create_instruction_dataloader` | function | `arclm.instruction_dataset.create_instruction_dataloader` | `(instructions, responses, tokenizer, block_size, batch_size, shuffle=True)` | Create DataLoader for instruction tuning. |
| `InstructionDataset` | class | `arclm.instruction_dataset.InstructionDataset` | `class` | Dataset for instruction-following: loss only on response tokens. |
| `disable_debug` | constant/object | `arclm.disable_debug` | `` |  |
| `enable_debug` | constant/object | `arclm.enable_debug` | `` |  |
| `get_device` | function | `arclm.get_device` | `()` | Return the preferred local torch device. |
| `get_version` | function | `arclm.get_version` | `()` | Return the installed ArcLM version. |
| `list_available_models` | function | `arclm.list_available_models` | `(models_dir: str \| Path='models')` | List local ``.pth`` checkpoints in a models directory. |
| `load_training_checkpoint` | function | `arclm.load_training_checkpoint` | `(path: str \| Path, device_type: str='cpu')` | Load a trusted ArcLM/PyTorch training checkpoint. |
| `report_environment` | function | `arclm.report_environment` | `(models_dir: str \| Path='models') -> str` | Return a short human-readable runtime report. |
| `create_simple_interface_app` | function | `arclm.create_simple_interface_app` | `()` | Create the optional Flask app for ArcLM's simple web interface. |
| `run_simple_interface` | function | `arclm.run_simple_interface` | `(*args, **kwargs)` | Run ArcLM's optional simple web interface. |

## Public Methods On Exported Classes

| Class | Method | Signature | Summary |
| --- | --- | --- | --- |
| `ArcLM` | `forward` | `(self, idx)` |  |
| `ArcLM` | `get_num_parameters` | `(self)` | Get total number of parameters |
| `Lab` | `inspect` | `(self, target: Any \| None=None) -> dict[str, Any] \| DatasetInspection` | Inspect data or the latest Lab decisions. |
| `Lab` | `create` | `(self, *, task: str='causal-lm', size: str='small', data: Dataset \| None=None, **overrides: Any) -> Model` | Create a small native model with inspectable defaults. |
| `Lab` | `plan` | `(self, model: Model, data: Dataset, **options: Any) -> TrainingPlan` | Create a training plan. |
| `Lab` | `train` | `(self, model: Model, data: Dataset, debug: bool=False, **options: Any) -> dict[str, Any]` | Train a model through the shared high-level Trainer. |
| `Lab` | `pretrain` | `(self, data: str \| Dataset, *, size: str='small', debug: bool=False, **options: Any) -> Model` | Inspect data, create a model, train it, and return the model. |
| `ModelBundle` | `predict` | `(self, prompt: str, **generation_kwargs: Any) -> str` | Generate text from the loaded causal LM. |
| `ModelBundle` | `save` | `(self, output_dir: str \| Path) -> Path` | Save this bundle to a directory. |
| `Model` | `create` | `(cls, *, architecture: str='arclm-native', tokenizer: Tokenizer \| SentencePieceTokenizer \| None=None, runtime: Runtime \| None=None, **config_values: Any) -> 'Model'` | Create a native ArcLM model through vNext lifecycle semantics. |
| `Model` | `load` | `(cls, source: str \| Path, *, runtime: Runtime \| None=None) -> 'Model'` | Load an ArcLM `.arcmodel` artifact. |
| `Model` | `save` | `(self, path: str \| Path, *, layout: str='auto', shard_size: str \| int \| None=None, overwrite: bool=False) -> ArcModelArtifact` | Save this model as an ArcLM-native `.arcmodel` artifact. |
| `Model` | `generate` | `(self, prompt: str, *, max_new_tokens: int=20, temperature: float=0.0) -> str` | Generate text using the native model. |
| `Model` | `inspect` | `(self) -> dict[str, Any]` | Return a structured vNext model report. |
| `Model` | `get_tensor` | `(self, name: str)` | Return a model tensor by state-dict name. |
| `Model` | `set_tensor` | `(self, name: str, value: Any) -> None` | Replace one tensor in the model state dict. |
| `Model` | `attach_lora` | `(self, *, rank: int=4, alpha: float=8.0, target_modules: tuple[str, ...] \| None=None) -> tuple[str, ...]` | Attach a native LoRA adapter to this model. |
| `Model` | `save_adapter` | `(self, path: str \| Path, *, overwrite: bool=False) -> Any` | Save attached native LoRA tensors as `.arcadapter`. |
| `Model` | `load_adapter` | `(self, path: str \| Path) -> Any` | Attach and load a native `.arcadapter`. |
| `Model` | `finetune` | `(self, *, data: Any, method: str='lora', **kwargs: Any) -> dict[str, Any]` | Fine-tune this model through the public Trainer. |
| `Dataset` | `load` | `(cls, source: str \| Path \| Iterable[dict[str, Any]], *, format: str \| None=None) -> 'Dataset'` | Load a local dataset source. |
| `Dataset` | `inspect` | `(self) -> DatasetInspection` | Return a lightweight dataset report. |
| `Dataset` | `prepare` | `(self, *, tokenizer: Tokenizer \| None=None, max_vocab: int=50000, block_size: int=8, batch_size: int=2, shuffle: bool=True) -> PreparedDataset` | Tokenize and batch the dataset for native causal LM training. |
| `Dataset` | `text` | `(self) -> str` | Return records as training text. |
| `ModelRegistry` | `register` | `(cls, spec: ModelSpec) -> None` | Register or replace a model spec. |
| `ModelRegistry` | `get` | `(cls, architecture_id: str) -> ModelSpec` | Return a registered spec by ID. |
| `ModelRegistry` | `resolve` | `(cls, config: dict[str, Any] \| str) -> ModelSpec` | Resolve a config or architecture name to a registered spec. |
| `ModelRegistry` | `supported` | `(cls) -> list[dict[str, Any]]` | Return all registered specs as support reports. |
| `ModelRegistry` | `supports` | `(cls, model: dict[str, Any] \| str, capabilities: Iterable[str] \| None=None) -> dict[str, str]` | Return capability support for a model/config. |
| `ModelSpec` | `supports` | `(self, capability: str) -> str` | Return the support level for one capability. |
| `ModelSpec` | `to_dict` | `(self) -> dict[str, Any]` | Return a JSON-safe spec report. |
| `Runtime` | `auto` | `(cls, *, prefer: str='auto', precision: str='auto') -> 'Runtime'` | Inspect the local machine and choose a runtime. |
| `Runtime` | `device_name` | `(self) -> str` | Return a backend device string for internal backend adapters. |
| `Runtime` | `torch_device` | `(self)` | Return a `torch.device` for low-level interoperability. |
| `Runtime` | `to_dict` | `(self) -> dict[str, Any]` | Return a JSON-safe runtime report. |
| `ModelSupportReport` | `is_supported` | `(self) -> bool` | Return whether ArcLM can attempt the requested workflow. |
| `ModelSupportReport` | `summary` | `(self) -> str` | Return a compact human-readable summary. |
| `ModelSupportReport` | `to_dict` | `(self) -> Dict[str, Any]` | Return a JSON-serializable representation. |
| `ModelSupportReport` | `raise_for_unsupported` | `(self) -> None` | Raise when the inspected source is unsupported. |
| `Config` | `to_dict` | `(self)` | Convert config to dictionary |
| `Config` | `load_config_from_model` | `(self, model_path)` |  |
| `Config` | `get_device` | `(self)` |  |
| `Config` | `validate` | `(self)` | Validate and normalize important configuration values in place. |
| `Config` | `set_safe` | `(self, **kwargs)` | Set config attributes safely, ignoring unknown keys. |
| `ArcLMConfig` | `to_dict` | `(self, *, redact: bool=True) -> dict[str, Any]` |  |
| `ArcLMConfig` | `to_workflow_dict` | `(self) -> dict[str, Any]` | Return a dict compatible with the existing workflow runner. |
| `ConfigMigrationReport` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `CheckpointInspectionReport` | `is_loadable` | `(self) -> bool` | Return whether the checkpoint looks loadable. |
| `CheckpointInspectionReport` | `is_verified` | `(self) -> bool` | Return whether integrity checks passed. |
| `CheckpointInspectionReport` | `summary` | `(self) -> str` | Return a compact human-readable summary. |
| `CheckpointInspectionReport` | `to_dict` | `(self) -> Dict[str, Any]` | Return a JSON-serializable report. |
| `CheckpointInspectionReport` | `raise_for_errors` | `(self) -> None` | Raise :class:`CheckpointError` when inspection found errors. |
| `DoctorReport` | `is_valid` | `(self) -> bool` |  |
| `DoctorReport` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `DoctorReport` | `to_json` | `(self) -> str` |  |
| `LoadingPolicy` | `safe` | `(cls) -> 'LoadingPolicy'` |  |
| `LoadingPolicy` | `trusted_local` | `(cls) -> 'LoadingPolicy'` |  |
| `LoadingPolicy` | `legacy_unsafe` | `(cls) -> 'LoadingPolicy'` |  |
| `LoadingPolicy` | `from_value` | `(cls, value: str \| 'LoadingPolicy') -> 'LoadingPolicy'` |  |
| `LoadingPolicy` | `to_dict` | `(self) -> dict[str, object]` |  |
| `DeviceConfig` | `resolve` | `(self) -> DeviceSelection` | Validate and resolve the requested runtime device. |
| `DeviceConfig` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `DeviceSelection` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `ResourceLimits` | `check_record` | `(self, record: dict[str, Any], index: int) -> None` |  |
| `ResourceLimits` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `CacheStats` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `Tokenizer` | `build` | `(self, text)` | Build vocabulary from text |
| `Tokenizer` | `encode` | `(self, tokens)` | Encode tokens to indices |
| `Tokenizer` | `encode_text` | `(self, text)` | Encode raw text to token indices. |
| `Tokenizer` | `decode` | `(self, indices)` | Decode indices back to tokens |
| `Tokenizer` | `decode_string` | `(self, indices)` | Decode indices to string |
| `Tokenizer` | `get_vocab_size` | `(self)` | Get vocabulary size |
| `Tokenizer` | `get_unknown_index` | `(self)` | Get the index used for unknown tokens. |
| `Tokenizer` | `get_rare_tokens` | `(self, min_count=2, limit=20)` | Return training tokens that occur fewer than min_count times. |
| `Tokenizer` | `analyze_coverage` | `(self, tokens, rare_threshold=2, top_unknown=20)` | Measure unknown-token and vocabulary coverage for a token sequence. |
| `Tokenizer` | `save_vocab` | `(self, path)` | Save vocabulary to file |
| `Tokenizer` | `load_vocab` | `(self, path)` | Load vocabulary from file |
| `Tokenizer` | `to_checkpoint` | `(self)` | Return tokenizer metadata that can be stored in a model checkpoint. |
| `Tokenizer` | `to_json` | `(self)` | Serialize tokenizer to JSON-compatible dict. |
| `Tokenizer` | `save` | `(self, path)` | Save tokenizer to a JSON file. |
| `Tokenizer` | `from_json` | `(cls, data)` | Load a word tokenizer from JSON-compatible dict. |
| `SentencePieceTokenizer` | `build` | `(self, text: str) -> dict` | Train a SentencePiece tokenizer from raw text. |
| `SentencePieceTokenizer` | `to_checkpoint` | `(self)` | Return tokenizer metadata that can be stored in a model checkpoint. |
| `SentencePieceTokenizer` | `from_checkpoint` | `(cls, metadata)` | Rebuild a SentencePiece tokenizer from checkpoint metadata. |
| `SentencePieceTokenizer` | `encode` | `(self, tokens)` | Encode tokens by joining them back into text first. |
| `SentencePieceTokenizer` | `encode_text` | `(self, text, out_type=int)` | Encode raw text to SentencePiece token IDs. |
| `SentencePieceTokenizer` | `decode` | `(self, indices)` | Decode IDs to SentencePiece pieces. |
| `SentencePieceTokenizer` | `decode_string` | `(self, indices)` | Decode IDs directly back to text. |
| `SentencePieceTokenizer` | `get_vocab_size` | `(self)` | Get vocabulary size. |
| `SentencePieceTokenizer` | `get_unknown_index` | `(self)` | Get the index used for unknown pieces. |
| `SentencePieceTokenizer` | `get_rare_tokens` | `(self, min_count=2, limit=20)` | Return rare whitespace tokens from the training text. |
| `SentencePieceTokenizer` | `to_json` | `(self)` | Serialize SentencePiece tokenizer. |
| `SentencePieceTokenizer` | `save` | `(self, path)` | Save SentencePiece tokenizer to a file. |
| `SentencePieceTokenizer` | `from_json` | `(cls, data)` | Load SentencePiece tokenizer from JSON-compatible dict. |
| `SentencePieceTokenizer` | `load` | `(cls, path)` |  |
| `SentencePieceTokenizer` | `analyze_coverage` | `(self, tokens, rare_threshold=2, top_unknown=20)` | Measure unknown-piece usage for a token sequence. |
| `SentencePieceTokenizer` | `get_meta_data` | `(self)` | Return metadata for the tokenizer. |
| `TokenizerFactory` | `register` | `(cls, tokenizer_type, tokenizer_class)` | Register a tokenizer class by type name. |
| `TokenizerFactory` | `create` | `(cls, tokenizer_type='word', **kwargs)` | Create a tokenizer by type name. |
| `TokenizerFactory` | `from_config` | `(cls, config)` | Create a tokenizer from a Config-like object. |
| `DataBundle` | `vocab_size` | `(self)` |  |
| `DataBundle` | `count` | `(self)` |  |
| `DataBundle` | `save_tokenizer` | `(self, path)` |  |
| `DatasetSource` | `close` | `(self) -> None` | Release source resources if a custom cleanup callback was supplied. |
| `DatasetSource` | `streaming` | `(self) -> bool` |  |
| `DatasetSource` | `seekable` | `(self) -> bool` |  |
| `DatasetSource` | `one_shot` | `(self) -> bool` |  |
| `DatasetSource` | `to_list` | `(self, limit: Optional[int]=None) -> list[Record]` | Materialize records, optionally limiting the count. |
| `DatasetSource` | `from_records` | `(cls, records: Iterable[Mapping[str, Any]]) -> 'DatasetSource'` |  |
| `DatasetSource` | `from_iterator` | `(cls, records: Iterable[Mapping[str, Any]], *, one_shot: bool=True) -> 'DatasetSource'` |  |
| `DatasetSource` | `from_jsonl` | `(cls, path: str \| Path, *, streaming: bool=True, encoding: str='utf-8', blank_lines: str='skip', malformed: str='raise') -> 'DatasetSource'` |  |
| `DatasetSource` | `from_text` | `(cls, path: str \| Path, *, streaming: bool=True, encoding: str='utf-8', blank_lines: str='skip') -> 'DatasetSource'` |  |
| `DatasetSource` | `from_csv` | `(cls, path: str \| Path, *, streaming: bool=True, encoding: str='utf-8', malformed: str='raise') -> 'DatasetSource'` |  |
| `DatasetSource` | `from_directory` | `(cls, path: str \| Path, *, format: Optional[str]=None, encoding: str='utf-8', malformed: str='raise') -> 'DatasetSource'` |  |
| `DatasetSource` | `from_huggingface` | `(cls, path: str, *, split: str='train', streaming: bool=True, **kwargs: Any) -> 'DatasetSource'` | Open a Hugging Face dataset lazily when ``datasets`` is installed. |
| `DataPipeline` | `inspect` | `(self) -> List[Dict[str, Any]]` | Return configured operation metadata before execution. |
| `DataPipeline` | `to_config` | `(self) -> List[Dict[str, Any]]` | Return serializable operation configuration where possible. |
| `DataPipeline` | `strip_whitespace` | `(self, field: str='text') -> 'DataPipeline'` | Strip leading/trailing whitespace from a string field. |
| `DataPipeline` | `normalize_text` | `(self, field: str='text') -> 'DataPipeline'` | Normalize repeated whitespace in a string field. |
| `DataPipeline` | `remove_empty` | `(self, field: str='text') -> 'DataPipeline'` | Remove records whose selected field is empty. |
| `DataPipeline` | `rename_fields` | `(self, mapping: Mapping[str, str]) -> 'DataPipeline'` | Rename fields using ``old_name -> new_name`` mapping. |
| `DataPipeline` | `select_fields` | `(self, fields: Sequence[str]) -> 'DataPipeline'` | Keep only selected fields. |
| `DataPipeline` | `join_fields` | `(self, fields: Sequence[str], output_field: str='text', separator: str='\n') -> 'DataPipeline'` | Join selected fields into one output field. |
| `DataPipeline` | `map_records` | `(self, function: RecordCallable, *, name: str='map_records') -> 'DataPipeline'` | Apply a custom record transformation. |
| `DataPipeline` | `filter_records` | `(self, predicate: FilterCallable, *, name: str='filter_records') -> 'DataPipeline'` | Keep records for which a predicate returns true. |
| `DataPipeline` | `deduplicate` | `(self, field: str='text') -> 'DataPipeline'` | Remove duplicate records using the selected field. |
| `DataPipeline` | `validate` | `(self, schema: str, *, strict: bool=True, allow_empty: bool=False) -> 'DataPipeline'` | Validate records and keep records unchanged. |
| `DataPipeline` | `format_prompt_completion` | `(self, output_field: str='text') -> 'DataPipeline'` | Format prompt/completion rows into a text field. |
| `DataPipeline` | `format_instruction` | `(self, output_field: str='text') -> 'DataPipeline'` | Format instruction/input/output rows into a text field. |
| `DataPipeline` | `format_conversations` | `(self, output_field: str='text') -> 'DataPipeline'` | Format conversation messages into a text field. |
| `DataPipeline` | `apply_tokenizer` | `(self, tokenizer: Any, field: str='text', output_field: str='tokens') -> 'DataPipeline'` | Add tokenizer output to each record. |
| `DataPipeline` | `split` | `(self, train: float=0.8, validation: float=0.1, test: float=0.1) -> 'DataPipeline'` | Shuffle deterministically and return one record with split lists. |
| `DataPipeline` | `run` | `(self, records: Iterable[Mapping[str, Any]], *, copy_records: bool=True) -> Tuple[List[Record], DataPipelineReport]` | Execute the configured pipeline. |
| `DataPipelineReport` | `is_valid` | `(self) -> bool` | Return whether no validation failures or operation errors occurred. |
| `DataPipelineReport` | `summary` | `(self) -> str` | Return a compact human-readable report summary. |
| `DataPipelineReport` | `to_dict` | `(self) -> Dict[str, Any]` | Return a JSON-serializable report. |
| `DataQualityReport` | `summary` | `(self) -> str` |  |
| `DataQualityReport` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `DataQualityReport` | `to_json` | `(self) -> str` |  |
| `DataQualityReport` | `to_markdown` | `(self) -> str` |  |
| `DatasetValidationReport` | `is_valid` | `(self) -> bool` | Return whether all records were valid. |
| `DatasetValidationReport` | `error_count` | `(self) -> int` | Return the number of validation errors. |
| `DatasetValidationReport` | `warning_count` | `(self) -> int` | Return the number of validation warnings. |
| `DatasetValidationReport` | `error_categories` | `(self) -> Dict[str, int]` | Return counts by error category. |
| `DatasetValidationReport` | `summary` | `(self) -> str` | Return a compact human-readable summary. |
| `DatasetValidationReport` | `to_dict` | `(self) -> Dict[str, Any]` | Return a JSON-serializable report. |
| `DatasetValidationReport` | `raise_for_errors` | `(self) -> None` | Raise :class:`DatasetValidationError` when errors exist. |
| `DuplicateReport` | `has_duplicates` | `(self) -> bool` |  |
| `DuplicateReport` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `ShardResult` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `SplitResult` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `ValidationIssue` | `to_dict` | `(self) -> Dict[str, Any]` | Return the issue as a dictionary. |
| `TextRecord` | `validate` | `(cls, record: Mapping[str, Any], *, index: int=0, strict: bool=True, allow_empty: bool=False)` |  |
| `PromptCompletionRecord` | `validate` | `(cls, record: Mapping[str, Any], *, index: int=0, strict: bool=True, allow_empty: bool=False)` |  |
| `InstructionRecord` | `validate` | `(cls, record: Mapping[str, Any], *, index: int=0, strict: bool=True, allow_empty: bool=False)` |  |
| `ConversationRecord` | `to_dict` | `(self) -> Dict[str, Any]` | Convert to a plain dictionary. |
| `ConversationRecord` | `validate` | `(cls, record: Mapping[str, Any], *, index: int=0, strict: bool=True, allow_empty: bool=False)` |  |
| `DataProcessor` | `load` | `(path: Any, format: Optional[str]=None, loader: Optional[Callable[[Path], Iterable[Dict[str, Any]]]]=None) -> ProcessedDataset` |  |
| `ProcessedDataset` | `transform` | `(self, format: str='pretraining', mapping: Optional[Dict[str, str]]=None, template: Optional[str]=None, text_fields: Optional[Iterable[str]]=None, tokenizer: Any=None) -> 'ProcessedDataset'` |  |
| `ProcessedDataset` | `tokenize` | `(self, tokenizer: Any, text_key: str='text') -> 'ProcessedDataset'` |  |
| `ProcessedDataset` | `split` | `(self, train: float=0.8, validation: float=0.1, test: float=0.1, seed: int=42) -> Dict[str, List[Dict[str, Any]]]` |  |
| `ProcessedDataset` | `filter` | `(self, predicate: Callable[[Dict[str, Any]], bool]) -> 'ProcessedDataset'` |  |
| `ProcessedDataset` | `clean` | `(self, text_keys: Optional[Iterable[str]]=None) -> 'ProcessedDataset'` |  |
| `ProcessedDataset` | `map_batches` | `(self, function: Callable[[List[Dict[str, Any]]], Iterable[Dict[str, Any]]], batch_size: int=32) -> 'ProcessedDataset'` |  |
| `Trainer` | `make_plan` | `(self) -> TrainingPlan` | Create an inspectable plan without executing training. |
| `Trainer` | `inspect` | `(self) -> dict[str, Any]` | Return current trainer state. |
| `Trainer` | `train` | `(self, *args: Any, mode: str='pretrain', debug: bool=False, **kwargs: Any) -> Any` | Train using the ArcLM Training Engine. |
| `LegacyTrainer` | `train` | `(self, loader, epochs, val_loader=None, early_stopping_patience=None, min_delta=None, checkpoint_callback=None, checkpoint_epoch_interval=None, checkpoint_batch_interval=None)` | Train the model with optional validation and early stopping |
| `LegacyTrainer` | `unpack_batch` | `(self, batch)` | Normalize tuple and dict dataloader batches. |
| `LegacyTrainer` | `compute_loss` | `(self, logits, y, loss_mask=None)` | Compute next-token loss, optionally over masked label positions only. |
| `LegacyTrainer` | `restore_best_model` | `(self)` | Restore weights from the best validation loss, if available. |
| `LegacyTrainer` | `get_generalization_gap` | `(self)` | Get generalization gap (val_loss - train_loss) |
| `LegacyTrainer` | `freeze_layers` | `(self, pattern: str='blocks', verbose: bool=True) -> int` | Freeze layers matching pattern for finetuning. |
| `LegacyTrainer` | `unfreeze_layers` | `(self, pattern: str=None, verbose: bool=True) -> int` | Unfreeze all or specific layers. |
| `LegacyTrainer` | `get_frozen_layers_info` | `(self) -> dict` | Return info about frozen/trainable layers. |
| `LegacyTrainer` | `get_train_history` | `(self)` | Get training history |
| `LegacyTrainer` | `save` | `(self, config: Config, vocab=None, stoi=None, itos=None, tokenizer_metadata=None)` | Save model checkpoint |
| `LegacyTrainer` | `load` | `(self, model_path)` | Load model checkpoint |
| `LegacyTrainer` | `exists` | `(self, model_path)` | Check if model checkpoint exists |
| `Generator` | `generate` | `(self, start_text, max_new_tokens=20, temperature=1.0, repetition_penalty=1.0, top_k=None, top_p=None)` | Generate text from start text |
| `Generator` | `generate_string` | `(self, start_text, max_new_tokens=20, temperature=1.0, repetition_penalty=1.0, top_k=None, top_p=None)` | Generate text and return as string |
| `LoadedModel` | `predict` | `(self, input_text, max_new_tokens=80, temperature=0.9, repetition_penalty=1.2, top_k=None, top_p=0.9)` | Generate a prediction from raw input text. |
| `GenerationResult` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `TrainingConfig` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `TrainingReport` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `ModelCapability` | `to_dict` | `(self) -> dict` | Return a JSON-serializable representation. |
| `ExternalLoadedModel` | `predict` | `(self, prompt: PromptLike, **generation_kwargs: Any) -> str` | Generate a single prediction from a string prompt or chat messages. |
| `ExternalLoadedModel` | `generate` | `(self, prompt: PromptLike, **generation_kwargs: Any) -> str` | Generate text with either a native ArcLM or Hugging Face model. |
| `ExternalLoadedModel` | `chat` | `(self, messages: Sequence[Mapping[str, str]], **generation_kwargs: Any) -> str` | Generate a response from OpenAI-style chat messages. |
| `ExternalLoadedModel` | `batch_predict` | `(self, prompts: Sequence[PromptLike], **generation_kwargs: Any) -> List[str]` | Generate one prediction per prompt. |
| `ExternalLoadedModel` | `to` | `(self, device: Union[str, torch.device]) -> 'ExternalLoadedModel'` | Move the underlying model to a device when the backend supports it. |
| `ExternalLoadedModel` | `save` | `(self, output_dir: PathLike, save_config: Optional[ModelSaveConfig]=None) -> Optional[Path]` | Save or export this loaded model using ArcLM save settings. |
| `ExternalLoadedModel` | `save_pretrained` | `(self, output_dir: PathLike, save_config: Optional[ModelSaveConfig]=None) -> Optional[Path]` | Alias for :meth:`save` for Hugging Face-style user expectations. |
| `ExternalLoadedModel` | `export` | `(self, output_dir: PathLike, save_config: Optional[ModelSaveConfig]=None) -> Optional[Path]` | Alias for :meth:`save` for explicit export workflows. |
| `GenerationConfig` | `from_overrides` | `(cls, base: Optional['GenerationConfig']=None, **overrides: Any) -> 'GenerationConfig'` | Build a generation config by applying keyword overrides. |
| `ModelSourceInfo` | `to_dict` | `(self) -> Dict[str, Any]` | Return a JSON-serializable representation of the inspection result. |
| `ModelSourceInfo` | `format_report` | `(self) -> str` | Return a compact human-readable inspection report. |
| `ModelSourceInfo` | `report` | `(self) -> str` | Backward-friendly alias for :meth:`format_report`. |
| `UnifiedPipeline` | `build` | `(self, vocab_size: int) -> 'UnifiedPipeline'` | Build model and trainer. |
| `UnifiedPipeline` | `train` | `(self, train_loader, val_loader=None, num_epochs: Optional[int]=None) -> Dict[str, Any]` | Execute training with stopping criteria. |
| `UnifiedPipeline` | `save_checkpoint` | `(self, path: Optional[str]=None) -> Path` | Save model checkpoint with metadata. |
| `UnifiedPipeline` | `get_model` | `(self) -> ArcLM` | Get the trained model. |
| `PreTrainedModelLoader` | `load` | `(self, revision: str \| None=None, trust_remote_code: bool=False) -> tuple` | Load model and return (model, metadata). |
| `ModelAdapter` | `adapt_weights` | `(self, verbose: bool=True) -> Dict[str, int]` | Adapt source model weights to target model architecture. |
| `StoppingCriteria` | `should_stop_by_steps` | `(self, current_step: int) -> bool` | Check if max_steps exceeded. |
| `StoppingCriteria` | `should_stop_by_validation` | `(self, val_loss: float, best_val_loss: float, patience_counter: int) -> bool` | Check if early stopping criteria met. |
| `BaseTrainingPipeline` | `build` | `(self, vocab_size: int)` | Build the pipeline for a vocabulary size. |
| `BaseTrainingPipeline` | `train` | `(self, train_loader, val_loader=None, num_epochs=None)` | Train the pipeline. |
| `BaseTrainingPipeline` | `save_checkpoint` | `(self, path=None)` | Save pipeline state. |
| `BaseTrainingPipeline` | `get_model` | `(self)` | Return the current model. |
| `BaseModelLoader` | `load` | `(self)` | Load a model source. |
| `BaseModelAdapter` | `adapt_weights` | `(self, verbose: bool=True)` | Adapt source weights into the target model. |
| `LoadedCheckpoint` | `is_arclm_checkpoint` | `(self) -> bool` |  |
| `LoadedCheckpoint` | `require_state_dict` | `(self) -> Dict[str, torch.Tensor]` |  |
| `LoadPlan` | `apply_overrides` | `(self, overrides: Dict[str, Any]) -> None` |  |
| `LoadPlan` | `to_dict` | `(self) -> Dict[str, Any]` |  |
| `LoadPlan` | `format_report` | `(self) -> str` |  |
| `LoadPlan` | `report` | `(self) -> str` |  |
| `ModelInspector` | `can_inspect` | `(self, source: Any) -> bool` |  |
| `ModelInspector` | `inspect` | `(self, source: Any) -> LoadPlan` |  |
| `SmartLoader` | `register_inspector` | `(cls, inspector: ModelInspector) -> None` |  |
| `SmartLoader` | `inspect` | `(cls, source: Any, auto_detect: bool=True, **overrides: Any) -> LoadPlan` |  |
| `SmartLoader` | `load` | `(cls, source: Any, auto_detect: bool=True, map_location: str='cpu', **overrides: Any) -> LoadedCheckpoint` |  |
| `MetricsReport` | `to_dict` | `(self)` | Convert to dictionary for JSON serialization. |
| `FingerprintReport` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `Run` | `start` | `(self) -> None` |  |
| `Run` | `complete` | `(self) -> None` |  |
| `Run` | `fail` | `(self, message: str) -> None` |  |
| `Run` | `log_config` | `(self, config: Any, name: str='config') -> Path` |  |
| `Run` | `log_report` | `(self, report: Any, name: str) -> Path` |  |
| `Run` | `log_metric` | `(self, name: str, value: float, *, step: Optional[int]=None) -> None` |  |
| `Run` | `warn` | `(self, message: str) -> None` |  |
| `Run` | `save_artifact` | `(self, source: str \| Path, *, name: Optional[str]=None) -> Path` |  |
| `RunMetadata` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `TokenizationConfig` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `TokenizedDataset` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `WorkflowResult` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `WorkflowStageResult` | `to_dict` | `(self) -> dict[str, Any]` |  |
| `L1Regularization` | `compute_loss` | `(self, model)` | Compute L1 regularization loss |
| `L2Regularization` | `compute_loss` | `(self, model)` | Compute L2 regularization loss |
| `EarlyStopping` | `check` | `(self, val_loss)` | Check if training should stop |
| `EarlyStopping` | `reset` | `(self)` | Reset early stopping |
| `LearningRateScheduler` | `step` | `(self, epoch)` | Update learning rate for current epoch |
| `GeneralizationMonitor` | `update` | `(self, train_loss, val_loss)` | Update monitor with new losses |
| `GeneralizationMonitor` | `get_generalization_gap` | `(self)` | Get current generalization gap (val - train) |
| `GeneralizationMonitor` | `get_average_gap` | `(self, window=10)` | Get average generalization gap over window |
| `GeneralizationMonitor` | `is_overfitting` | `(self, threshold=0.5)` | Check if model is overfitting |
| `GeneralizationMonitor` | `get_report` | `(self)` | Get generalization report |
| `LabelSmoothing` | `forward` | `(self, pred, target)` | Compute smoothed cross-entropy loss |
