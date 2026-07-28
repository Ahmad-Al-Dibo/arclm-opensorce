# Threat Model

ArcLM assumes datasets, model repositories, caches, checkpoints, plugins, and
callbacks may be untrusted unless the user explicitly marks them trusted.

| Threat | Default Mitigation | Residual Risk |
| --- | --- | --- |
| Malicious pickle checkpoint | Safe checkpoint inspection rejects legacy pickle files. | Trusted-local and legacy-unsafe modes still depend on user judgment. |
| Malicious remote model code | `trust_remote_code=False` by default. | Users can explicitly opt in. |
| Checkpoint tampering | Manifest hash verification for format v1 checkpoints. | Legacy `.pt` files cannot be safely verified without deserialization. |
| Cache poisoning | Cache metadata must mark entries complete and corrupted entries are reported. | JSON cache data can still be intentionally wrong. |
| Secret leakage | Config export and security scans redact common token/password patterns. | Novel secret formats may not be detected. |
| Dataset-content leakage | Quality reports redact samples by default. | Users can opt into sample output. |
| Path traversal | Config paths are normalized; checkpoint inspection avoids archive extraction. | User-supplied writable directories still need OS permissions. |
| Denial-of-service records | Resource limits and malformed JSONL handling exist. | Full streaming backpressure remains future work. |
| Plugin/callback failure | Callback exceptions are captured as warnings. | Plugin code is still arbitrary local Python. |

Security tests use synthetic local fixtures only.
