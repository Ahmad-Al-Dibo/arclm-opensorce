# Fine-Tune A Causal Model

Native fine-tuning:

```python
from arclm import train_model

result = train_model(
    mode="finetune",
    data="domain.txt",
    output="domain-model.pth",
    checkpoint="base-model.pth",
    num_epochs=1,
)
```

Hugging Face SFT:

```python
from arclm import train_sft

result = train_sft(
    model="hf-internal-testing/tiny-random-gpt2",
    dataset="sft.jsonl",
    output_dir="outputs/sft",
    max_steps=1,
)
```

The Hugging Face example may download model files and requires optional dependencies.

