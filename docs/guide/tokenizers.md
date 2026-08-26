# Tokenizers

ArcLM exposes one public tokenizer abstraction:

```python
from arclm import Tokenizer

tokenizer = Tokenizer(strategy="word", max_vocab=16).build("alpha beta alpha")
ids = tokenizer.encode("alpha beta")
text = tokenizer.decode(ids)
```

## Current Strategies

- `word`: native whitespace tokenizer.
- `character`: native character tokenizer.
- `sentence`: optional SentencePiece-backed tokenizer loaded only when requested.

Aliases such as `char`, `sentencepiece`, `bpe`, and `unigram` resolve to current
engines where available. ArcLM-native BPE, WordPiece, and Unigram engines are
planned; they are not currently implemented as native engines.

## Research Tokenizers

Custom tokenizer engines can be registered:

```python
from arclm import Tokenizer

Tokenizer.register_engine("my-tokenizer", MyTokenizerEngine)
tokenizer = Tokenizer(strategy="my-tokenizer")
```

The engine must implement the `TokenizerEngine` contract exposed from
`arclm.research`.
