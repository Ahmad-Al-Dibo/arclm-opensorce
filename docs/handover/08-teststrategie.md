# Teststrategie

## Testlagen

De suite is klein maar breed:

- schema/data: `test_schemas.py`, `test_data_pipeline.py`,
  `test_data_processor.py`, `test_data_scale.py`
- model support/loading: `test_model_support_runtime.py`,
  `test_smart_loader.py`, `test_supported_models.py`
- certification: `test_gpt2_certification.py`, `test_llama_certification.py`,
  `test_gpu_certification.py`
- training/SFT: `test_instruction_sft.py`, `test_sft_api.py`,
  `test_checkpoint_reliability.py`
- ops/workflow: `test_runs_events_workflow.py`,
  `test_tokenization_cache.py`, `test_phase3_cli.py`
- docs/stability/security: `test_docs_inventory.py`,
  `test_phase4_hardening.py`,
  `test_extensibility_security_benchmarks.py`

## Markers

Markers in `pyproject.toml` zijn belangrijk voor releasecommunicatie:

- `unit`
- `integration`
- `cli`
- `docs`
- `security`
- `checkpoint`
- `config`
- `compatibility`
- `scale`
- `benchmark`
- `transformers`
- `hf`
- `slow`
- `cpu`
- `model_certification`
- `gpu`
- `gpu_certification`

Als een test bewijs levert voor een releaseclaim, markeer hem expliciet. Een
test die wel draait maar geen marker heeft, verdwijnt uit rapportages.

## Snelle veilige checks

Voor kleine docs/API wijzigingen:

```bash
python -m pytest tests/test_docs_inventory.py -q
python -m mkdocs build --strict
```

Voor codewijzigingen aan stabiele modules:

```bash
python -m compileall -q arclm tests
python -m ruff check .
python -m mypy arclm
python -m pytest -m "not slow"
```

Voor model support:

```bash
python -m pytest -m transformers
python -m pytest -m model_certification
arclm model inspect hf-internal-testing/tiny-random-gpt2 --json
```

## Testdata en netwerk

Tiny Hugging Face modellen kunnen netwerk/HF rate-limit gevoelig zijn. Tests
gebruiken `hf-internal-testing` artifacts om downloads klein te houden, maar ze
zijn nog steeds externe afhankelijkheden.

GPU tests moeten skippen als CUDA ontbreekt. Een skip is geen certificatie.
