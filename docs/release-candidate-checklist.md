# Release Candidate Checklist

Before tagging `0.9.0rc1`:

1. freeze stable API changes
2. run the full Python `3.9` to `3.12` CI matrix
3. run CPU model certification
4. run GPU certification on actual CUDA hardware or mark GPU untested
5. run configuration migration tests
6. run checkpoint security tests
7. run dependency audit and secret scan
8. build wheel and source distributions
9. generate artifact checksums
10. generate SBOM
11. scan package contents
12. install wheel in a clean environment
13. install source distribution in a clean environment
14. run `arclm doctor --json` from each clean environment
15. run acceptance workflows
16. review changelog and deprecations
17. publish to TestPyPI only after approval
18. install from TestPyPI and rerun smoke checks
19. tag `0.9.0rc1`
20. fix only release-blocking issues before final `0.9.0`
