# Current Status

## Architecture

ArcLM has a cleaned package structure with public API, architecture contracts,
tokenizers, datasets, training, fine-tuning, runtime, artifacts, and research
extension points separated.

## Model Independence

The core model path is ArcLM-owned. `Model.create(...)`, `Model.load(...)`, the
architecture registry, tokenizer facade, training, generation, and artifacts do
not use Hugging Face Transformers.

Transformers compatibility is planned as an optional adapter layer, not as the
framework center.

## Tokenizer

Implemented:

- one public `Tokenizer`
- native word tokenizer
- native character tokenizer
- optional lazy SentencePiece strategy
- custom engine registration

Planned:

- native BPE
- native WordPiece
- native Unigram

## Training

Implemented:

- ArcLM-owned training engine
- deterministic `steps_per_epoch`
- global `max_steps`
- validation
- checkpoints and resume
- callbacks/events
- progress display
- early stopping
- structured metrics

## Fine-Tuning

Implemented:

- full fine-tuning
- trainable/frozen parameter selection
- native LoRA-style adapter attachment
- `.arcadapter` export

## Artifacts

Implemented:

- `.arcmodel` save/load
- config/tokenizer/architecture metadata
- safetensors weights
- integrity hashes
- training metadata

Planned:

- richer format migrations
- broader multi-shard workflows
- additional model families
