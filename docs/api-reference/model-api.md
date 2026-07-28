# Model API

| API | Import path | Purpose | Parameters | Returns | Stability |
| --- | --- | --- | --- | --- | --- |
| `ArcLM` | `arclm.ArcLM` | Compact GPT-style causal LM. | `vocab_size`, `embed_dim`, `block_size`, `num_blocks`, `dropout` | PyTorch module | Stable-ish |
| `MiniGPT` | `arclm.MiniGPT` | Backward-compatible alias for `ArcLM`. | Same as `ArcLM` | PyTorch module | Legacy |
| `build_model` | `arclm.build_model` | Build native model from config. | `config`, `vocab_size=None` | `ArcLM` | Stable-ish |
| `ModelCapability` | `arclm.ModelCapability` | Model support metadata. | dataclass fields | Capability record | Stable-ish |
| `get_supported_models` | `arclm.get_supported_models` | List support records. | `status=None` | `list[ModelCapability]` | Stable-ish |
| `get_model_capability` | `arclm.get_model_capability` | Look up a family. | `family` | `ModelCapability` | Stable-ish |
| `is_model_officially_supported` | `arclm.is_model_officially_supported` | Check official status. | `family` | `bool` | Stable-ish |
| `ModelSupportReport` | `arclm.ModelSupportReport` | Runtime model support report. | dataclass fields | Report | Stable-ish |
| `inspect_model_support` | `arclm.models.inspect_model_support` / `arclm.inspect_model_support` | Inspect config/tokenizer/task support before full load. | `source`, `task`, `device`, `precision`, `trust_remote_code`, `tokenizer_path` | `ModelSupportReport` | Stable-ish |
| `ModelBundle` | `arclm.ModelBundle` | Consolidated loaded model bundle. | dataclass fields | Bundle | Stable-ish |
| `arclm.models.load_model` | `arclm.models.load_model` | Load native or HF causal-LM sources through the new facade. | `source`, `task`, `device`, `precision`, `trust_remote_code`, `local_files_only` | `ModelBundle` | Stable-ish |
| `inspect_model_source` | `arclm.inspect_model_source` | Inspect model path or ID. | `source`, `**kwargs` | `ModelSourceInfo` or legacy `LoadPlan` for legacy kwargs | Experimental |
| `SmartLoader` | `arclm.SmartLoader` | Inspect/load checkpoint-like sources. | class methods | `LoadedCheckpoint` | Experimental |
| `load_external_model` | `arclm.load_external_model` | Load checkpoints/state dicts. | `source`, `map_location` | `LoadedCheckpoint` | Experimental |
| `adapt_for_training` | `arclm.adapt_for_training` | Adapt loaded checkpoint to native model. | checkpoint/config/tokenizer flags | `AdaptedModelBundle` | Experimental |
