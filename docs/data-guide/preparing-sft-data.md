# Preparing Supervised Fine-Tuning Data

`train_sft` accepts JSON, JSONL, CSV, or TXT through `DataProcessor.load`, then normalizes rows into messages.

Supported row shapes include:

```json
{"instruction": "Say hi", "output": "Hi."}
```

```json
{"messages": [{"role": "user", "content": "Say hi"}, {"role": "assistant", "content": "Hi."}]}
```

Run:

```python
from arclm import train_sft

result = train_sft(
    model="hf-internal-testing/tiny-random-gpt2",
    dataset="sft.jsonl",
    output_dir="outputs/sft",
    max_steps=1,
)
```

This may download model files.

