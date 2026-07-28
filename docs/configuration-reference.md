# Configuration Reference

`Config` is a flexible object used by native training.

Common fields:

| Field | Default | Purpose |
| --- | --- | --- |
| `embed_dim` | `64` | Embedding size. |
| `block_size` | `8` | Context length. |
| `num_blocks` | `2` | Transformer block count. |
| `dropout` | `0.0` | Dropout probability. |
| `batch_size` | `64` | Batch size. |
| `num_epochs` | `100` | Epoch count. |
| `learning_rate` | `1e-3` | Optimizer learning rate. |
| `weight_decay` | `0.0` | AdamW weight decay. |
| `tokenizer_type` | `word` | `word` or `sentencepiece`. |
| `max_vocab` | `50000` | Tokenizer vocabulary cap. |
| `validation_split` | `0.0` | Validation split ratio. |
| `device` | `cpu` in `Config`, auto in `create_config` | Runtime device. |

Prefer `create_config(**kwargs)` when you want unknown-field validation.

