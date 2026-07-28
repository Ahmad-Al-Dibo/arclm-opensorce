# Roadmap

## Priority 1

- Make tests run in supported Python versions with valid Torch installations.
- Consolidate CLI training/evaluation/generation around the modern public APIs.
- Add formal dataset schema validation.
- Add automated docs build and link checks.

## Priority 2

- Certify GPT-2 or Qwen as the first external official model family.
- Add reproducible model-verification scripts.
- Add checkpoint resume tests.
- Add structured logging instead of direct prints.

## Priority 3

- Define deprecation policy for `MiniGPT`, legacy `pipeline_v2`, and unrelated `logics` exports.
- Improve type hints and docstring coverage across helpers.
- Add large-dataset streaming and memory-behavior documentation.

## Out Of Scope For Now

- Encoder-only model training.
- Seq2seq model training.
- RLHF, DPO, PPO, reward modeling.
- Vision, audio, and multimodal model workflows.

