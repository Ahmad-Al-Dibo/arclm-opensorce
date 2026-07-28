# Loading Data

Use `DataProcessor.load`.

```python
from arclm import DataProcessor

jsonl_data = DataProcessor.load("data.jsonl")
json_data = DataProcessor.load("data.json")
csv_data = DataProcessor.load("data.csv")
txt_data = DataProcessor.load("data.txt")
```

For custom loading:

```python
def loader(path):
    yield {"text": path.read_text(encoding="utf-8")}

dataset = DataProcessor.load("custom.txt", loader=loader)
```

