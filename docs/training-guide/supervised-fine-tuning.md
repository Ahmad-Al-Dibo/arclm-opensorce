# Supervised Fine-Tuning

Use `train_sft` for Hugging Face causal-LM SFT.

```python
from arclm import train_sft

result = train_sft(
    model="hf-internal-testing/tiny-random-gpt2",
    dataset="sft.jsonl",
    output_dir="outputs/sft",
    backend="huggingface",
    max_steps=1,
)
```

Only `backend="huggingface"` is implemented in `train_sft`. ArcLM-native instruction tuning remains available through `InstructionDataset` and `Trainer`.

