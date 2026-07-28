# Systeemkaart

## Mentale kaart

ArcLM is geen monolithische trainer maar een verzameling workflow-lagen rond
causal language modeling. De betrouwbare richting is:

`records -> schema/report -> pipeline/quality/split -> tokenization/cache -> model support -> load -> train/evaluate/generate -> run reports`

Er bestaan tegelijk oudere, gebruiksvriendelijke APIs naast nieuwere
productiegerichte APIs. Voor nieuwe code moet je de nieuwere APIs als
architectuurkern behandelen:

- Data: `arclm.schemas`, `arclm.data_pipeline`, `arclm.data_sources`,
  `arclm.data_quality`, `arclm.tokenization`
- Models: `arclm.models`, `arclm.supported_models`, `arclm.certification`
- Config/workflows: `arclm.config.ArcLMConfig`, `arclm.workflow`
- Ops/security: `arclm.checkpoints`, `arclm.security`, `arclm.runs`,
  `arclm.doctor`, `arclm.release`, `arclm.stability`

De oudere kern blijft belangrijk voor backward compatibility:

- `DataProcessor` en `ProcessedDataset` zijn in-memory en eenvoudig, maar niet
  de beste basis voor streaming of formele validatie.
- `Config` is legacy training-config; `ArcLMConfig` is het schema-versioned
  workflowmodel.
- `arclm.inference.load_model` laadt native ArcLM checkpoints; `arclm.models.load_model`
  is de geconsolideerde facade voor native plus Hugging Face causal LMs.
- `external_inference.py` bevat veel historische externe-model functionaliteit
  en heeft andere defaults dan de nieuwe facade.

## Belangrijkste verborgen grens

De projectnaam suggereert "LLM library", maar de echte eigen modelarchitectuur
is compact en educatief: single-head causal self-attention, kleine GPT-style
blocks, geen multi-head aandacht en geen production-scale transformer stack.
De productie-orientatie zit vooral in workflow-betrouwbaarheid,
data-validatie, expliciete supportniveaus, checkpoint-inspectie en reports.

## Waar data en state landen

- Workflow runs schrijven naar `runs/<timestamp>_<name>_<id>/`.
- Tokenization cache gebruikt een dataset fingerprint plus tokenization config.
- Native training schrijft meestal legacy `.pth` checkpoints.
- Nieuwe checkpoint-inspectie verwacht voor veilige directory checkpoints een
  `manifest.json`, `hashes.json`, `model/config.json`,
  `model/model.safetensors`, en optionele training/tokenizer folders.
- `site/`, `dist/`, `build/`, `.arclm/`, `runs/`, venvs en caches zijn generated
  en horen niet in release-artifacts.

## Public surface met risico

`arclm.__init__` re-exporteert veel symbolen, inclusief historische logic
helpers en legacy loaders. Dit is bewust backward-compatible, maar het maakt de
top-level namespace geen zuivere architectuurkaart. Gebruik `arclm.stability`
als bron voor wat stabiel/provisional/deprecated is.
