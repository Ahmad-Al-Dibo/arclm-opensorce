# Tokenize A Dataset

```python
from arclm import DataProcessor, Tokenizer

dataset = DataProcessor.load("data.txt").clean().transform(format="pretraining")
tokenizer = Tokenizer(max_vocab=1000)
tokenizer.build(" ".join(row["text"] for row in dataset.samples))
tokenized = dataset.tokenize(tokenizer)

print(tokenized.samples[0]["tokens"])
```

