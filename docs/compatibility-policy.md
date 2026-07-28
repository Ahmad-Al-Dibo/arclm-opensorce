# Compatibility Policy

For the `0.9.x` line:

- stable Python APIs listed by `arclm.stability.stable_api_paths()` require
  explicit snapshot updates before incompatible changes
- provisional APIs may change in minor development releases
- experimental APIs may change when certification evidence changes
- deprecated APIs emit warnings when used and identify replacements
- configuration schema version `1` is the only current typed schema
- checkpoint format version `1` is inspectable without loading model weights
- cache reports use schema version `1.0`
- CLI commands in `arclm.stability.cli_manifest()` have separate stability labels

Tested local environment for this phase: Windows, Python `3.12`, CPU PyTorch.
Linux is covered by CI configuration. macOS and GPU behavior remain unverified
unless a separate run reports otherwise.
