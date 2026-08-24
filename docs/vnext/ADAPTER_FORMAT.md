# ArcLM Adapter Format

Status: IMPLEMENTED FOR NATIVE LoRA.

`.arcadapter` is an ArcLM-native directory artifact:

```text
adapter.arcadapter/
  manifest.json
  adapter.safetensors
```

## Manifest

The manifest records:

| Field | Meaning |
| --- | --- |
| `format` | `arcadapter`. |
| `format_version` | Adapter container version. |
| `adapter_type` | Currently `lora`. |
| `base_model_id` | Artifact/base identifier expected by the adapter. |
| `base_model_fingerprint` | Hash of base tensors excluding adapter tensors. |
| `architecture_id` | Expected registered architecture. |
| `target_modules` | ArcLM module names wrapped by LoRA. |
| `rank` | LoRA rank. |
| `alpha` | LoRA alpha. |
| `adapter_tensors` | Expected adapter tensor names, dtypes and shapes. |
| `tensor_file` | Safetensors payload path. |
| `tensor_hash` | SHA-256 hash of the safetensors payload. |
| `created_with` | ArcLM version string. |

## Compatibility

`Model.load_adapter()` validates format, architecture id, base model id, base
fingerprint, tensor hash, tensor names and tensor shapes. Incompatible adapters
raise `ModelCompatibilityError`; corrupted adapter tensors raise
`ArtifactIntegrityError`.
