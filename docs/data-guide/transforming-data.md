# Transforming Data

`ProcessedDataset.transform` creates a `text` field.

```python
dataset = DataProcessor.load("qa.jsonl").transform(
    format="instruction",
    mapping={"instruction": "question", "output": "answer"},
    template="Question: {instruction}\nAnswer: {output}",
)
```

Supported built-in `format` values are `pretraining`, `instruction`, and `chat`.

