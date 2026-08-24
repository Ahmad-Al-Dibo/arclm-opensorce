# ArcLM Pretrained Loading

Status: IMPLEMENTED FOR `arclm-native-causal-lm` ARTIFACTS.

The implemented pretrained/base family is ArcLM native causal-LM artifacts. This
was selected because it is the only model family already represented by native
ArcLM architecture code in this repository.

```text
Model.load("base.arcmodel")
  -> ArcModelArtifact manifest/config/tokenizer inspection
  -> ModelRegistry.resolve()/get()
  -> ModelSpec
  -> Architecture.build()
  -> WeightReader
  -> WeightMapper
  -> Runtime
  -> ArcLM Model
```

## Weight Mapping

`arclm-native-causal-lm` uses an identity weight mapper because saved native
tensor names are already ArcLM-owned names:

```text
blocks.0.attn.query.weight -> blocks.0.attn.query.weight
```

Family-specific mappings belong on `ModelSpec`. Generic loading code does not
contain model-family string heuristics for the native path.

## External Families

Existing GPT-2/Qwen/Llama/Hugging Face loaders remain transitional compatibility
code outside this Core pipeline. They have not been promoted into native ArcLM
architecture builders in this phase.
