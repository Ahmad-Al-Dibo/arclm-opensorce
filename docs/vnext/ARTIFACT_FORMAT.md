# ArcLM Artifact Format

Status: IMPLEMENTED FOR NATIVE V1.

ArcLM owns the `.arcmodel` container, manifest, config, tensor index, metadata
and integrity checks. Tensor bytes are encoded with safetensors.

## Layouts

Single portable file:

```text
model.arcmodel
```

Internally this is a zip archive containing:

```text
manifest.json
config.json
tokenizer.json
weights/model.safetensors
```

Directory:

```text
model/
  manifest.json
  config.json
  tokenizer.json
  weights/model.safetensors
```

Sharded:

```text
model/
  manifest.json
  config.json
  tokenizer.json
  weights/
    index.json
    shard-00001.safetensors
    shard-00002.safetensors
    ...
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
| `weights_hash` | SHA-256 hash of the stored weight file, or combined shard hash for sharded layouts. |
| `tokenizer_hash` | SHA-256 hash of `tokenizer.json`. |
| `layout` | `single`, `directory` or `sharded`. |
| `weight_files` | Relative safetensors files. |
| `tensor_metadata` | Expected tensor names, dtypes and shapes. |
| `tensor_index_hash` | SHA-256 hash of `weights/index.json` for sharded layouts. |
| `metadata` | JSON metadata owned by ArcLM. |

## Shard Index

`weights/index.json` is versioned:

```json
{
  "format": "arcweights-index",
  "format_version": "1",
  "schema_version": "1",
  "encoding": "safetensors",
  "max_shard_size_bytes": 1048576,
  "weight_map": {
    "blocks.0.attn.query.weight": "shard-00001.safetensors"
  },
  "shards": [
    {
      "file": "shard-00001.safetensors",
      "hash": "...",
      "bytes": 1234,
      "tensors": ["blocks.0.attn.query.weight"]
    }
  ]
}
```

## Integrity

`Model.load()` validates manifest hashes, tokenizer hash, shard-index hash,
per-shard hashes, expected tensor names and tensor shapes. Corrupt shard tests
raise `ArtifactIntegrityError`.
