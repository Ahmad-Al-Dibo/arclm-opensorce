# ArcLM Artifact Format

Status: EXPERIMENTAL.

The first vNext `.arcmodel` layout is a directory:

```text
model.arcmodel/
  manifest.json
  config.json
  tokenizer.json
  weights.pt
```

## Manifest Fields

| Field | Meaning |
| --- | --- |
| `artifact_id` | Stable generated artifact identifier. |
| `format` | Currently `arcmodel`. |
| `format_version` | Container format version, currently `1`. |
| `schema_version` | Manifest schema version, currently `1`. |
| `minimum_reader_version` | Minimum ArcLM reader version for this artifact. |
| `architecture_id` | Registered architecture ID, currently `arclm-native-causal-lm`. |
| `created_at` | UTC timestamp. |
| `dtype` | Weight dtype summary. |
| `weights_file` | Relative weight file path. |
| `weights_hash` | SHA-256 hash of the stored weights file. |
| `tokenizer_hash` | SHA-256 hash of `tokenizer.json`. |
| `metadata` | JSON metadata owned by ArcLM. |

The tensor bytes are currently stored with Torch serialization as an internal
implementation detail. Future versions may switch the weight file to
Safetensors without changing the high-level `Model.save()` / `Model.load()`
contract.
