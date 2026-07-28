# Tokenization

ArcLM includes:

- `Tokenizer`: whitespace word tokenizer.
- `SentencePieceTokenizer`: SentencePiece BPE/unigram tokenizer.
- `TokenizerFactory`, `create_tokenizer`, `get_tokenizer_from_config`.

Native checkpoints can store tokenizer metadata through `Trainer.save()` and `train_model()`.

