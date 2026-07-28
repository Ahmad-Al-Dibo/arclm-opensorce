# Tokenization API

| API | Import path | Purpose | Parameters | Returns | Stability |
| --- | --- | --- | --- | --- | --- |
| `Tokenizer` | `arclm.Tokenizer` | Word tokenizer. | `max_vocab`, `default_token`, `user_defined_symbols` | Tokenizer | Stable-ish |
| `Tokenizer.build` | `arclm.Tokenizer.build` | Build vocabulary from text. | `text` | `None` | Stable-ish |
| `Tokenizer.encode_text` | `arclm.Tokenizer.encode_text` | Encode raw text. | `text` | `list[int]` | Stable-ish |
| `Tokenizer.decode_string` | `arclm.Tokenizer.decode_string` | Decode IDs to text. | `indices` | `str` | Stable-ish |
| `SentencePieceTokenizer` | `arclm.SentencePieceTokenizer` | SentencePiece tokenizer. | `max_vocab`, `model_type`, `character_coverage`, `user_defined_symbols` | Tokenizer | Stable-ish |
| `TokenizerFactory` | `arclm.TokenizerFactory` | Registry for tokenizers. | class methods | tokenizer | Stable-ish |
| `create_tokenizer` | `arclm.create_tokenizer` | Create tokenizer by type. | `tokenizer_type`, `**kwargs` | tokenizer | Stable-ish |
| `get_tokenizer_from_config` | `arclm.get_tokenizer_from_config` | Build tokenizer from config. | `config` | tokenizer | Stable-ish |
| `tokenizer_from_checkpoint` | `arclm.tokenizer_from_checkpoint` | Restore tokenizer from native checkpoint. | `checkpoint_or_source` | tokenizer | Stable-ish |

Raises: tokenizer methods raise `ValueError` when used before `build()` or when metadata is missing.

