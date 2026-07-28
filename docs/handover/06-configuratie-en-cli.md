# Configuratie en CLI

## Twee configsystemen

`Config` is legacy native training config. Het accepteert historisch veel
kwargs, heeft defaults voor training, tokenizer, diagnostics en finetuning, en
wordt gebruikt door `train_model`, `prepare_data`, `Trainer` en native
inference.

`ArcLMConfig` is het nieuwe schema-versioned workflowmodel. Deze route heeft:

- `schema_version`
- typed dataclasses per sectie
- strict/permissive unknown field beleid
- padnormalisatie
- expliciete env expansion via `allow_env=True`
- secret redaction in `to_dict(redact=True)`
- `to_workflow_dict()` adapter naar de bestaande workflow runner

Nieuwe config-gedreven features moeten in `ArcLMConfig` landen en daarna naar
legacy structs adapteren waar nodig.

## Workflow runner

`run_workflow` accepteert JSON/TOML of dict. Als `schema_version` aanwezig is,
wordt de config via `load_arclm_config(..., permissive=True)` geladen en daarna
naar het interne workflowdict vertaald. Zonder `schema_version` blijft legacy
workflowconfig werken.

Stages staan hard-coded in volgorde:

`load`, `validate`, `quality`, `deduplicate`, `split`, `tokenize`, `model`,
`train`, `evaluate`

Een nieuwe stage toevoegen betekent minimaal:

- stage naam toevoegen aan default lijst
- `_run_stage` branch
- context contract documenteren
- report schrijven via `run.log_report`
- CLI dry-run gedrag testen

## CLI exit codes

De CLI gebruikt expliciete codes:

- `0` success
- `2` invalid usage
- `3` config error
- `4` dataset validation
- `5` unsupported model
- `6` model load
- `7` training
- `8` checkpoint
- `9` optional dependency missing
- `10` partial workflow

Nieuwe CLI commands moeten geen raw tracebacks tonen voor normale user errors,
maar wel JSON output ondersteunen als het om inspectie/validatie gaat.

## CLI drift risico

De CLI is beter geconsolideerd dan vroeger, maar sommige routes gebruiken nog
legacy internals:

- `arclm train` gebruikt `create_config` + `train_model`
- `arclm evaluate` gebruikt native `inference.load_model`
- `arclm generate` gebruikt native `inference.load_model`
- `arclm model inspect` gebruikt de nieuwe support facade

Als je externe HF generatie via CLI uitbreidt, gebruik `arclm.models.load_model`
en niet blind de native inference loader.
