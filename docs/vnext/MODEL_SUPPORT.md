# ArcLM vNext Model Support

Status: IMPLEMENTED FOR ONE NATIVE FAMILY.

| Model | Architecture ID | Inference | Pretraining | Full FT | LoRA | CPU | CUDA | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ArcLM native causal LM artifacts | `arclm-native-causal-lm` | SUPPORTED | SUPPORTED | EXPERIMENTAL | SUPPORTED | SUPPORTED | UNTESTED | Chosen because it is the only architecture already implemented natively in this repository; registry, ModelSpec, weight mapper, generation, safetensors artifact and native LoRA are tested. |
| Hugging Face causal LM | legacy external IDs | TRANSITIONAL | UNSUPPORTED | TRANSITIONAL | TRANSITIONAL | UNTESTED | UNTESTED | Existing Transformers loaders remain outside vNext Core. |
| PEFT adapter | legacy PEFT folders | TRANSITIONAL | UNSUPPORTED | UNSUPPORTED | TRANSITIONAL | UNTESTED | UNTESTED | Existing PEFT path remains compatibility code; native `.arcadapter` is implemented for ArcLM native LoRA. |
