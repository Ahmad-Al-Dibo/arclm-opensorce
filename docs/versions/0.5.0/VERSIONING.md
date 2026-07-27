# ArcLM Versioning And Release Guide

ArcLM 0.5.0 is the public baseline release described by this repository.

## Version Source Of Truth

The package version lives in:

```text
arclm/_version.py
```

`pyproject.toml`, `arclm.__version__`, and `arclm.get_version()` read from that file.

Check the installed version with:

```bash
python -c "import arclm; print(arclm.__version__)"
```

For this release, the expected value is:

```text
0.5.0
```

## Versioning Policy

ArcLM follows semantic versioning:

```text
MAJOR.MINOR.PATCH
```

- `PATCH`: bug fixes and documentation corrections.
- `MINOR`: compatible public features.
- `MAJOR`: breaking public API changes.

While ArcLM remains pre-1.0, treat every public API change carefully and document migration notes.

## 0.5.0 Release Scope

The 0.5.0 baseline documents:

- native `ArcLM` model training through `train_model()`;
- native next-token fine-tuning and continued training;
- Hugging Face SFT through `train_sft(backend="huggingface")`;
- assistant-only label masking;
- optional PEFT LoRA for Hugging Face SFT;
- native `InstructionDataset` and `Trainer` extension points;
- checkpoint loading, SmartLoader inspection, diagnostics, preprocessing, and examples.

Do not reference superseded post-baseline release numbers in public docs for this release.

## Release Command

Build and check the current release:

```bash
python setup.py --version
python setup.py release --version 0.5.0
```

Upload only when ready:

```bash
python setup.py release --version 0.5.0 --upload
```

PyPI does not allow replacing files that already exist for the same version. If a version file has already been uploaded, choose a new unpublished version rather than trying to overwrite it.

## Manual Release Checklist

1. Confirm `arclm/_version.py` matches the intended release.
2. Confirm `README.md` renders correctly on GitHub and PyPI.
3. Confirm `CHANGELOG.md` includes the release notes.
4. Run `python -m pytest tests`.
5. Run `python -m build`.
6. Run `python -m twine check dist/*`.
7. Inspect `git status` and commit only intentional changes.
8. Tag the release when the build is final.

Example tag for the baseline release:

```bash
git tag -a v0.5.0 -m "ArcLM 0.5.0"
git push origin v0.5.0
```

## Documentation Rule

Any public behavior change must update the matching documentation and runnable examples:

- public functions/classes;
- config options;
- dataset formats;
- examples;
- trainer behavior;
- checkpoint behavior;
- masking behavior;
- loader behavior;
- SFT/LoRA behavior.

Do not document planned features as implemented.
