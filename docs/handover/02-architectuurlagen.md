# Architectuurlagen

## Laag 1: record- en datasetcontracten

`arclm.schemas` definieert de formele recordvormen: `text`,
`prompt_completion`, `instruction`, en `conversation`. Deze laag verzamelt
fouten in `DatasetValidationReport` in plaats van records stil te repareren.
Dat is een belangrijke productiekeuze: filtering en reparatie moeten expliciet
via pipeline-operaties gebeuren.

`strict=True` behandelt onbekende velden als fouten. `strict=False` degradeert
ze naar warnings. Metadata is optioneel en moet een dict zijn.

## Laag 2: data-operaties

Er zijn twee generaties:

- `DataProcessor`/`ProcessedDataset`: snelle in-memory convenience voor JSON,
  JSONL, CSV, TXT, cleaning, transform, tokenize en split.
- `DatasetSource`/`DataPipeline`/`data_quality`: productiegerichter,
  deterministic, streaming of repeatable waar mogelijk, structured reports.

Nieuwe features moeten op de tweede generatie landen. Gebruik `DataProcessor`
alleen als compatibility/convenience wrapper.

## Laag 3: tokenization en cache

`tokenize_dataset` materialiseert momenteel het iterable naar een lijst voordat
het tokenizet. Dat is bewust simpel maar betekent dat "streaming source" niet
automatisch "streaming tokenization" is. De cache key bevat dataset fingerprint,
tokenization config en ArcLM-versie. Cache invalidatie is dus conservatief bij
versiewijzigingen.

## Laag 4: model support en loading

`supported_models.py` is declaratieve supportmetadata. `models.inspect_model_support`
verbindt die metadata met runtime-inspectie door veilig config/tokenizer info te
laden voordat volledige gewichten geladen worden.

`models.load_model` is de gewenste facade. Die valideert causal-LM support en
houdt `trust_remote_code=False` tenzij de gebruiker expliciet iets anders kiest.

## Laag 5: training/evaluatie/inference

Native training loopt nog via `pipeline.train_model`, `Trainer`, `Config`,
`prepare_data` en native checkpoints. Nieuwe workflowtraining gebruikt in
`workflow.py` de `training.train` facade, maar veel echte native features zitten
nog in `pipeline.py`.

Evaluation en generation zijn structured-report-first geworden:
`EvaluationReport`, `GenerationResult`, `WorkflowResult`.

## Laag 6: operationele laag

`Run` legt configs, reports, metrics en metadata vast. `doctor` inspecteert de
runtime zonder modellen te downloaden. `release` scant artifacts, checksums en
SBOM. `security` centraliseert loading policy en secret scanning.

Deze laag is bedoeld om open-source gebruikers simpele "kan ik dit veilig
draaien?" tooling te geven zonder enterprise platformfeatures te bouwen.
