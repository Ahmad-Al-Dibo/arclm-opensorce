# Security en release

## Threat model in code

ArcLM moet vooral beschermen tegen:

- onveilige PyTorch pickle checkpoints
- impliciete remote code execution via Hugging Face
- accidental secrets in docs/tests/releases
- release artifacts met generated/private bestanden
- stille mismatch tussen model, tokenizer en config

## LoadingPolicy

`LoadingPolicy` heeft drie modes:

- `safe`: geen pickle, geen remote code, hashes verifieren
- `trusted_local`: pickle toegestaan voor lokale trusted artifacts, geen remote code
- `legacy_unsafe`: pickle en remote code toegestaan, hash verification uit

`legacy_unsafe` geeft een runtime warning. Gebruik die mode niet in tests behalve
om de warning expliciet te bewijzen.

## Checkpoint security

Safe inspection van legacy `.pt`, `.pth`, `.ckpt`, `.bin` laadt niets. Dat is
bewust, ook als het betekent dat een bestaand checkpoint "error" krijgt in safe
mode. De juiste UX is: inspecteer veilig, vraag expliciet vertrouwen, laad pas
dan.

Directory checkpoints met `manifest.json` kunnen zonder weight deserialisatie
worden gecontroleerd. Hash mismatch en missing files moeten errors blijven.

## Remote code

Nieuwe code mag `trust_remote_code` nooit automatisch op `True` zetten. Als een
model remote code nodig heeft, moet dat zichtbaar zijn in support report,
config, CLI flag of loading policy.

## Release helpers

`arclm.release` bevat:

- `scan_distribution`
- `artifact_checksums`
- `generate_sbom`

De scanner zoekt naar generated/private patronen zoals venvs, `site/`,
`build/`, `.arclm`, `runs/`, caches en modelgewichten. Dit is geen volledige
supply-chain scanner, maar een snelle RC-gate.

## Dependency security

`pip-audit` kan lokale editable packages en PyTorch CPU wheels overslaan omdat
die niet normaal op PyPI gemapt zijn. Documenteer zulke skips altijd los van
"geen vulnerabilities".
