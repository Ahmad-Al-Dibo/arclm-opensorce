# Release Readiness

Phase 4 moves the development line to `0.9.0`.

Implemented release-candidate foundations:

- public API stability manifest and snapshot tests
- schema-versioned typed configuration
- configuration migration reports
- safe checkpoint inspection and hash verification
- explicit checkpoint loading policies
- device and precision validation
- `arclm doctor`
- release artifact checksums, content scanning, and SBOM helpers
- model-certification protocol scaffolding
- experimental Llama-family CPU certification using `hf-internal-testing/tiny-random-LlamaForCausalLM`

The open-source framework is ready for focused hardening and release-candidate
testing around the certified CPU workflows. GPU certification, strict typing
across the whole legacy package, additional fully certified model families, and
resume-equivalence evidence remain gates for broader compatibility claims.
