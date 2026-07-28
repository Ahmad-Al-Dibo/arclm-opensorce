# Playbooks voor veilige wijzigingen

## Nieuwe data-operatie toevoegen

1. Voeg de operatie toe aan `DataPipeline`.
2. Zorg dat input records standaard niet muteren.
3. Geef een stabiele operation name en serialiseerbare params.
4. Rapporteer affected/removed/errors.
5. Voeg unit tests toe op deterministic output, reportinhoud en copy gedrag.
6. Voeg CLI alleen toe als er een duidelijke user workflow is.

## Nieuw recordtype toevoegen

1. Voeg schema toe aan `VALID_SCHEMAS`.
2. Maak een dataclass met `allowed_fields`, `required_fields`, `validate`.
3. Voeg formatting toe in tokenization als het naar causal-LM tekst moet.
4. Update `validate_records` dispatch en stats.
5. Voeg strict/permissive tests toe.
6. Documenteer unknown-field en empty-field gedrag.

## Nieuw modelgezin ondersteunen

1. Voeg capability metadata toe in `supported_models.py` met conservatieve
   status.
2. Voeg detectie toe in `models.inspect_model_support`.
3. Certificeer eerst met tiny artifact.
4. Test config load, tokenizer load, model load, causal LM detectie,
   deterministic generation, eval, minimal SFT, save/reload.
5. Pas status pas aan naar `official` als de test in CI of releaseprocedure
   bewijs levert.

## CLI command toevoegen

1. Implementeer businesslogica eerst als Python API.
2. CLI command moet alleen argumenten mappen en output/exit code regelen.
3. Voeg `--json` toe voor inspectie/validatie commands.
4. Gebruik bestaande exit code categorieen.
5. Voeg help-test en minstens een gedragstest toe.
6. Voeg CLI stability toe in `arclm.stability`.

## Configveld toevoegen

1. Voeg veld toe aan de juiste typed config dataclass.
2. Valideer in `__post_init__`.
3. Voeg toe aan redacted serialization als het secret-achtig is.
4. Voeg migratie toe als legacy equivalent bestaat.
5. Update `to_workflow_dict` als de workflow runner het gebruikt.
6. Test strict unknown fields en permissive mode.

## Checkpointformaat uitbreiden

1. Breid manifest uit zonder bestaande v1 readers te breken.
2. Houd safe inspection weight-load-vrij.
3. Voeg hash coverage toe voor nieuwe files.
4. Maak missing file errors expliciet.
5. Test tamper, missing, unexpected en version mismatch.
6. Documenteer trust mode als pickle nodig is.

## Bugfix triage

Gebruik deze volgorde:

1. Reproduceer via kleinste Python API test.
2. Check of CLI dezelfde codepad gebruikt; zo niet, beslis facade vs legacy.
3. Voeg regressietest toe bij de laag waar het contract hoort.
4. Pas docs alleen aan als gebruikersgedrag verandert.
5. Run marker subset plus `pytest -m "not slow"` voor cross-component risico.
