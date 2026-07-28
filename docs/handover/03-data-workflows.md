# Data-workflows

## Belangrijkste businesslogica

ArcLM behandelt datasetkwaliteit als eerste klas onderdeel van het framework.
De businessregel is: ongeldige data wordt gerapporteerd, niet stil verwijderd.
Pas `DataPipeline.remove_empty`, `filter_records`, `deduplicate` of custom
`map_records` toe als expliciete herstelstap.

## Schemas

De vier schemas zijn hard-coded in `VALID_SCHEMAS`:

- `text`: vereist `text: str`
- `prompt_completion`: vereist `prompt: str`, `completion: str`
- `instruction`: vereist `instruction: str`, `output: str`; `input` mag leeg
- `conversation`: vereist `messages`, ieder item met role/content

Conversation roles zijn beperkt tot `system`, `user`, `assistant`. Dit is een
verborgen modelaanname: ArcLM formatteert gesprekken generiek als tekstregels,
niet met model-specifieke chat templates.

## DataPipeline-regels

`DataPipeline.run` kopieert records standaard. Dit beschermt user input tegen
mutatie en moet behouden blijven. Callable operaties zijn toegestaan, maar niet
volledig serialiseerbaar; ze krijgen `serializable=False` in operation metadata.

Elke operatie moet een stabiele naam hebben en output leveren in de vorm:

`records, removed_count, affected_count, warnings, errors`

Als je een nieuwe pipeline-operatie toevoegt, voeg direct toe:

- operation metadata via `_add`
- deterministic gedrag bij gelijk seed/config
- reportvelden voor affected/removed/errors
- unit test op "input wordt niet gemuteerd"
- CLI of docs alleen als het echt publieke workflowwaarde heeft

## Streaming valkuil

`DatasetSource` kan streaming zijn, maar veel downstream functies materialiseren
naar `list[dict]`: validation, split, shard, tokenization en workflow stages.
Noem een nieuwe feature dus niet automatisch "large-scale streaming" tenzij de
hele route lazy blijft.

## Split/shard verborgen aannames

`split_dataset` gebruikt standaard hash-splitting op de volledige record JSON.
Kleine datasets kunnen lege splits krijgen; dat wordt als warning gerapporteerd.
Gebruik `group_key` voor leakage-preventie op user/conversation/document niveau.

`shard_dataset` ondersteunt `contiguous`, `round_robin`, `hash`. Hash gebruikt
seed plus geselecteerde key of stable JSON. Dit maakt sharding reproduceerbaar,
maar schemawijzigingen veranderen hash-uitkomsten.

## DataProcessor als legacy/convenience

`DataProcessor.load` is handig maar minder streng:

- JSON scalars worden `{"text": value}`
- JSON dicts met `data`, `samples` of `records` worden uitgepakt
- TXT slaat lege regels over
- geen schema-validatie tijdens load

Nieuwe features moeten `DataProcessor` niet uitbreiden als er al een plek in
`DatasetSource`, `schemas`, `DataPipeline` of `data_quality` bestaat.
