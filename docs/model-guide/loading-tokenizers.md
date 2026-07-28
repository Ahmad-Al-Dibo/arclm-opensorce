# Loading Tokenizers

Native tokenizers:

```python
from arclm import Tokenizer, SentencePieceTokenizer

tok = Tokenizer(max_vocab=1000)
tok.build("some training text")
tok.save("tokenizer.json")
```

Restore from native checkpoint:

```python
from arclm import tokenizer_from_checkpoint

tokenizer = tokenizer_from_checkpoint("model.pth")
```

Hugging Face tokenizers are loaded internally by `load_any_model` and `train_sft` with `AutoTokenizer.from_pretrained`.

