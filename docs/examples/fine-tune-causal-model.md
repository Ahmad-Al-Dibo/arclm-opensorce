# Fine-Tune A Causal Model

Native fine-tuning:

```python
from arclm import Dataset, Model, Runtime, Trainer

runtime = Runtime.auto(prefer="cpu")
model = Model.load("base-model.arcmodel", runtime=runtime)
dataset = Dataset.load("domain.txt")

history = Trainer(
    model=model,
    dataset=dataset,
    epochs=1,
    batch_size=2,
    learning_rate=5e-4,
).train(mode="full_finetune")

model.save("domain-model.arcmodel", overwrite=True)
print(history["train_losses"])
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
