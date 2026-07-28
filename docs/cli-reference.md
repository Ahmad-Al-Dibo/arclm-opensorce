# CLI Reference

ArcLM can be run as a module or installed console script:

```bash
python -m arclm --help
arclm --help
```

## Commands

```bash
arclm version
arclm info
arclm doctor --json
arclm config validate arclm.json
arclm config show arclm.json
arclm config migrate old.json --output arclm.json
arclm data inspect data.jsonl --json
arclm data validate data.jsonl --schema text --strict --json
arclm data prepare data.jsonl --output clean.jsonl --schema text
arclm model inspect hf-internal-testing/tiny-random-gpt2 --json
arclm model list
arclm model load-check hf-internal-testing/tiny-random-gpt2
arclm model certify hf-internal-testing/tiny-random-gpt2 --family gpt2 --json
arclm checkpoint inspect runs/example/checkpoint --json
arclm checkpoint verify runs/example/checkpoint
arclm train --data data.txt --output models/model.pth
arclm evaluate --model models/model.pth --data test.txt
arclm eval --model models/model.pth --data test.txt
arclm generate --model models/model.pth --prompt "The future is"
python -m arclm --run simple-interface
```

## Notes

`arclm train` now delegates to `train_model` and no longer exposes the unused `--num-heads` option. `arclm eval` remains as an alias for `arclm evaluate`.

## Preprocessing CLI

The preprocessing module also has a module entry point:

```bash
python -m arclm.preprocess.cli raw.jsonl --output clean.jsonl --report-dir reports/preprocess
```
