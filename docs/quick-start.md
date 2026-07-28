# Quick Start

This example creates a tiny dataset, cleans and validates it, tokenizes it, trains a native ArcLM causal model, and runs inference.

```python
from pathlib import Path
import tempfile

from arclm import DataProcessor, Tokenizer, load_model, train_model

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    raw_path = root / "records.jsonl"
    train_path = root / "train.txt"
    model_path = root / "model.pth"

    raw_path.write_text(
        '{"text": "ArcLM prepares language model data."}\n'
        '{"text": "Clean records make training easier."}\n',
        encoding="utf-8",
    )

    dataset = (
        DataProcessor.load(raw_path)
        .clean()
        .filter(lambda row: isinstance(row.get("text"), str) and len(row["text"]) > 10)
        .transform(format="pretraining")
    )
    if not dataset.samples:
        raise ValueError("No valid training rows remain.")

    tokenizer = Tokenizer(max_vocab=64)
    tokenizer.build(" ".join(row["text"] for row in dataset.samples))
    tokenized = dataset.tokenize(tokenizer)
    assert all("tokens" in row for row in tokenized.samples)

    train_path.write_text(
        (" ".join(row["text"] for row in dataset.samples) + " ") * 24,
        encoding="utf-8",
    )

    train_model(
        mode="pretrain",
        data=str(train_path),
        output=str(model_path),
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
    print(loaded.predict("ArcLM", max_new_tokens=4, top_k=3))
```

Expected output is a saved checkpoint and a short generated string. The model is intentionally tiny, so generation quality is not meaningful.

