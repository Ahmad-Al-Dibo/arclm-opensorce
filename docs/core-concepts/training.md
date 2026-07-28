# Training

Use `train_model` for native ArcLM workflows:

```python
from arclm import train_model

result = train_model(mode="pretrain", data="data.txt", output="model.pth", num_epochs=1)
```

Modes: `pretrain`, `finetune`, `continue_training`.

Use `train_sft` for Hugging Face supervised fine-tuning.

