# Configuration API

| API | Import path | Purpose | Parameters | Returns | Stability |
| --- | --- | --- | --- | --- | --- |
| `Config` | `arclm.Config` | Central native training config. | keyword fields | Config | Stable-ish |
| `create_config` | `arclm.create_config` | Create config with unknown-key validation. | `**kwargs` | `Config` | Stable-ish |
| `get_finetuning_config` | `arclm.config.get_finetuning_config` | Fine-tuning defaults. | common training fields | `Config` | Stable-ish |
| `get_instruction_tuning_config` | `arclm.config.get_instruction_tuning_config` | Instruction-tuning defaults. | common training fields | `Config` | Stable-ish |
| `load_config_yaml` | `arclm.config_loader.load_config_yaml` | Read YAML config. | `filepath` | `Config` | Stable-ish |
| `load_config_json` | `arclm.config_loader.load_config_json` | Read JSON config. | `filepath` | `Config` | Stable-ish |
| `save_config_yaml` | `arclm.config_loader.save_config_yaml` | Write YAML config. | `config`, `filepath` | `None` | Stable-ish |
| `save_config_json` | `arclm.config_loader.save_config_json` | Write JSON config. | `config`, `filepath` | `None` | Stable-ish |
| `ExternalModelConfig` | `arclm.ExternalModelConfig` | External loading config. | dataclass fields | Config | Experimental |
| `ModelSaveConfig` | `arclm.ModelSaveConfig` | Model save/export settings. | dataclass fields | Config | Experimental |
| `PreprocessConfig` | `arclm.preprocess.PreprocessConfig` | JSONL preprocessing config. | dataclass fields | Config | Experimental |
| `normalize_device` | `arclm.normalize_device` | Validate and normalize device values. | `device` | `str` | Stable-ish |
| `normalize_precision` | `arclm.normalize_precision` | Validate and normalize precision values. | `precision` | `str` | Stable-ish |
| `validate_training_config` | `arclm.validate_training_config` | Validate native training config in place. | `config` | config | Stable-ish |
