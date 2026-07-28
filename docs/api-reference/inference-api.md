# Inference API

| API | Import path | Purpose | Parameters | Returns | Stability |
| --- | --- | --- | --- | --- | --- |
| `load_model` | `arclm.load_model` | Load native checkpoint. | `model_path`, `device`, `prefer_best` | `LoadedModel` | Stable-ish |
| `LoadedModel` | `arclm.LoadedModel` | Native inference wrapper. | dataclass fields | Wrapper | Stable-ish |
| `predict` | `arclm.predict` | Load native checkpoint and predict. | `input_text`, `model_path`, `device`, `reload`, generation options | `str` | Stable-ish |
| `load_any_model` | `arclm.load_any_model` | Load native or external source. | `source`, load options | `ExternalLoadedModel` | Experimental |
| `ExternalLoadedModel` | `arclm.ExternalLoadedModel` | Unified external/native wrapper. | dataclass fields | Wrapper | Experimental |
| `GenerationConfig` | `arclm.GenerationConfig` | External generation defaults. | dataclass fields | Config | Experimental |
| `predict_external` | `arclm.predict_external` | Load source and generate once. | `source`, `prompt`, options | `str` | Experimental |
| `load_external_for_inference` | `arclm.load_external_for_inference` | Alias for `load_any_model`. | `source`, options | `ExternalLoadedModel` | Experimental |

