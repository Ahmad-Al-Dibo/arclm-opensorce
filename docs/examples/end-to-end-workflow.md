# Complete End-To-End Workflow

```python
from pathlib import Path
import tempfile

from arclm import DataProcessor, Tokenizer, load_model, train_model

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    raw = root / "records.jsonl"
    train = root / "train.txt"
    model_path = root / "model.pth"

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
    dataset = dataset.tokenize(tokenizer)

    train.write_text((" ".join(row["text"] for row in dataset.samples) + " ") * 24, encoding="utf-8")

    train_model(
        "pretrain",
        str(train),
        str(model_path),
        tokenizer_type="word",
        max_vocab=64,
        embed_dim=16,
        num_blocks=1,
        block_size=8,
        batch_size=2,
        num_epochs=1,
        validation_split=0.0,
        training_log_interval=0,
        device="cpu",
    )

    loaded = load_model(model_path, device="cpu")
    print(loaded.predict("ArcLM", max_new_tokens=4))
```

