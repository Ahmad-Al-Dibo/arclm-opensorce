# Complete End-To-End Workflow

```python
from pathlib import Path
import tempfile

from arclm import DataProcessor, Lab, Model, Runtime, Tokenizer

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    raw = root / "records.jsonl"
    train = root / "train.txt"
    artifact = root / "model.arcmodel"

    raw.write_text(
        '{"text": "ArcLM loads data."}\n'
        '{"text": "ArcLM trains causal models."}\n',
        encoding="utf-8",
    )

    dataset = (
        DataProcessor.load(raw)
        .clean()
        .filter(lambda row: len(row["text"]) >= 10)
        .transform(format="pretraining")
    )

    tokenizer = Tokenizer(max_vocab=64)
    tokenizer.build(" ".join(row["text"] for row in dataset.samples))
    tokenized = dataset.tokenize(tokenizer)

    train.write_text((" ".join(row["text"] for row in tokenized.samples) + " ") * 24, encoding="utf-8")

    runtime = Runtime.auto(prefer="cpu")
    lab = Lab(runtime=runtime)
    model = lab.pretrain(train, size="tiny", epochs=1, learning_rate=1e-3)
    model.save(artifact, overwrite=True)

    loaded = Model.load(artifact, runtime=runtime)
    print(loaded.generate("ArcLM", max_new_tokens=4))
```
