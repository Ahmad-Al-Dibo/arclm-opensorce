# Training, checkpoints en runs

## Native modelarchitectuur

`ArcLM` is compact:

- token embedding + positional embedding
- `num_blocks` GPT-style blocks
- single-head causal self-attention
- feed-forward met GELU
- LayerNorm en linear head

Er is geen echte multi-head attention. Oude CLI/config verwijzingen naar
`num_heads` mogen niet opnieuw als werkende feature gedocumenteerd worden.

## Native trainingroute

`train_model` in `pipeline.py` is de echte high-level native route. Die bouwt:

1. legacy `Config`
2. data via `prepare_data`
3. tokenizer of checkpoint-tokenizer
4. model via `build_model` of adapter
5. `Trainer`
6. checkpoint save via `Trainer.save`

`continue_training` vereist een checkpoint met restorable tokenizer metadata of
een expliciete tokenizer. Dit is belangrijker dan het lijkt: zonder dezelfde
tokenizer is optimizer/model resume semantisch waardeloos.

## Trainer valkuilen

`Trainer` bevat nog `print()` progress output naast async metrics logging. Voor
library-achtige nieuwe code liever package logging gebruiken, maar bestaande
prints niet agressief verwijderen zonder CLI UX-test.

Early stopping wordt genegeerd als er geen `val_loader` is. Lege loaders geven
warnings of stoppen stil na validatie. Dit gedrag is user-facing geworden door
tests; verander het alleen met migratienote.

## Checkpoints

Er zijn twee checkpointwerelden:

- Legacy native `.pth`: PyTorch pickle, nodig voor bestaande users, niet veilig
  voor onbekende bronnen.
- Nieuwe directory checkpoint inspectie: manifest + hashes + safetensors-first.

`inspect_checkpoint(path, trust="safe")` hoort de default voor tooling te zijn.
`load_trusted_checkpoint` mag alleen voor trusted local legacy files.

## Runs

`Run` schrijft direct naar disk:

- `run.json`
- `config/*.json`
- `reports/*.json`
- `metrics/metrics.jsonl`
- `artifacts/*`

Writes zijn simpel en niet overal atomisch. Als je failure recovery uitbreidt,
begin bij `_write_json`, cache writes en checkpoint manifest writes.

## Resume betrouwbaarheid

Er bestaan tests voor save/load en minimale resume-achtige paden, maar nog geen
sterke "uninterrupted vs resumed training geeft equivalent resultaat" garantie.
Beschouw resume dus als belangrijk uitbreidingspunt voor een productieclaim.
