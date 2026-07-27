# Custom Trainers And Real Usage Examples

This guide shows how to use ArcLM below the high-level helpers, and how to
write your own trainer when `train_model()` or `train_sft()` is not enough.

Use this file with:

- `arclm/trainer.py`: the native training loop implementation.
- `train.py`: a complete Qwen medical SFT script with dataset preparation and
  periodic Hugging Face checkpoint folders.
- `test.py`: a matching loader/generation script for the adapter or full model
  saved by `train.py`.

## Which API Should I Use?

| Goal | Recommended path |
| --- | --- |
| Train a small native ArcLM model from text | `train_model(mode="pretrain")` |
| Fine-tune a native ArcLM checkpoint with next-token loss | `train_model(mode="finetune")` |
| Continue native pretraining with strict checkpoint compatibility | `train_model(mode="continue_training")` |
| Fine-tune a Hugging Face causal LM with SFT | `train_sft(backend="huggingface")` |
| Train native ArcLM with assistant-only labels | `InstructionDataset` + `Trainer` |
| Add custom loss, logging, callbacks, or batch handling | subclass `Trainer` or write a loop around `Trainer` helpers |
| Reproduce the repository Qwen medical workflow | root `train.py` and `test.py` |

`train_model()` and `train_sft()` are the shortest stable public entry points.
The root `train.py` script uses advanced helpers from `arclm.sft` so it can add
detailed checkpoint folders and dataset analysis around the Hugging Face SFT
backend. That script is a practical application workflow, not the smallest API
surface.

## The Native Trainer Contract

`Trainer` trains native `ArcLM` models. It expects:

- a model whose `forward(x)` returns logits with shape `[batch, time, vocab]`;
- labels `y` with shape `[batch, time]`;
- optional `mask` with shape `[batch, time]`, where non-zero positions
  contribute to the loss;
- a config with fields such as `device`, `learning_rate`, `num_epochs`,
  `grad_clip`, `checkpoint_interval`, and `checkpoint_batch_interval`.

Accepted batch formats:

```python
(x, y)
```

or:

```python
{"x": x, "y": y, "mask": optional_loss_mask}
```

The dict format is what `InstructionDataset` uses for assistant-only native
SFT. Plain `TextDataset` and `create_dataloader()` return tuple batches.

## Minimal Low-level Native Training

This is the smallest fully manual native ArcLM flow. It mirrors what
`train_model(mode="pretrain")` does internally, but leaves each part visible.

```python
from pathlib import Path

from arclm import (
    Tokenizer,
    build_model,
    build_trainer,
    create_config,
    create_dataloader,
)

text = Path("data/data.txt").read_text(encoding="utf-8")

tokenizer = Tokenizer(max_vocab=2000)
tokenizer.build(text)
encoded = tokenizer.encode_text(text)

config = create_config(
    vocab_size=tokenizer.get_vocab_size(),
    block_size=64,
    batch_size=8,
    embed_dim=128,
    num_blocks=2,
    learning_rate=3e-4,
    num_epochs=2,
    model_path="models/custom_native_arclm.pth",
    training_log_interval=10,
    device="cpu",
)

loader = create_dataloader(
    encoded_data=encoded,
    block_size=config.block_size,
    batch_size=config.batch_size,
    shuffle=True,
)

model = build_model(config)
trainer = build_trainer(model, config)
trainer.train(loader, config.num_epochs)
trainer.save(
    config,
    vocab=tokenizer.vocab,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    tokenizer_metadata=tokenizer.to_checkpoint(),
)
```

Keep `len(encoded) > block_size`; otherwise `TextDataset` cannot make any
next-token training examples.

## Validation And Early Stopping

`Trainer.train()` supports validation loaders and early stopping:

```python
from arclm import create_dataloader

split = int(len(encoded) * 0.9)
train_ids = encoded[:split]
val_ids = encoded[split:]

train_loader = create_dataloader(train_ids, config.block_size, config.batch_size)
val_loader = create_dataloader(
    val_ids,
    config.block_size,
    config.batch_size,
    shuffle=False,
)

trainer.train(
    train_loader,
    config.num_epochs,
    val_loader=val_loader,
    early_stopping_patience=3,
    min_delta=1e-4,
)
```

When validation is available, `Trainer` tracks:

- `train_losses`
- `val_losses`
- `val_perplexities`
- `val_token_accuracies`
- best validation weights
- current epoch, batch, and global step

Use:

```python
history = trainer.get_train_history()
```

## Native Assistant-only SFT

Use `InstructionDataset` when training the built-in ArcLM model on
instruction/response pairs. It builds a loss mask so only response tokens are
trained.

```python
from torch.utils.data import DataLoader

from arclm import Config, InstructionDataset, Tokenizer, build_model, build_trainer

instructions = [
    "Explain SFT in one sentence.",
    "What does assistant-only loss mean?",
]
responses = [
    "SFT trains a model on examples of desired responses.",
    "It computes loss only on assistant answer tokens.",
]

training_text = "\n".join(
    f"<|instruction|>\n{i}\n<|response|>\n{r}"
    for i, r in zip(instructions, responses)
)

tokenizer = Tokenizer(
    max_vocab=1000,
    user_defined_symbols=["<|instruction|>", "<|response|>"],
)
tokenizer.build(training_text)

config = Config(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=64,
    block_size=64,
    num_blocks=2,
    batch_size=2,
    num_epochs=1,
    learning_rate=5e-4,
    model_path="models/native_instruction_sft.pth",
    device="cpu",
)

dataset = InstructionDataset(
    instructions=instructions,
    responses=responses,
    tokenizer=tokenizer,
    block_size=config.block_size,
)

loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

model = build_model(config)
trainer = build_trainer(model, config)
trainer.train(loader, config.num_epochs)
trainer.save(
    config,
    vocab=tokenizer.vocab,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    tokenizer_metadata=tokenizer.to_checkpoint(),
)
```

## Subclass `Trainer` For A Custom Loss

Override `compute_loss()` when your model output and batch format still match
ArcLM, but the loss needs to change.

```python
import torch.nn.functional as F

from arclm import Trainer


class LabelSmoothingTrainer(Trainer):
    def __init__(self, *args, label_smoothing=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, logits, y, loss_mask=None):
        batch_size, steps, vocab_size = logits.shape
        loss_per_token = F.cross_entropy(
            logits.reshape(batch_size * steps, vocab_size),
            y.reshape(batch_size * steps),
            reduction="none",
            label_smoothing=self.label_smoothing,
        ).reshape(batch_size, steps)

        if loss_mask is None:
            return loss_per_token.mean()

        loss_mask = loss_mask.to(loss_per_token.device, dtype=loss_per_token.dtype)
        return (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
```

Create it the same way as the normal trainer:

```python
import torch

model = build_model(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

trainer = LabelSmoothingTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    config=config,
    label_smoothing=0.05,
)
trainer.train(loader, config.num_epochs)
```

## Write A Manual Loop Around Trainer Helpers

If you need full control, you can still reuse `Trainer.unpack_batch()` and
`Trainer.compute_loss()`:

```python
model.train()

for epoch in range(config.num_epochs):
    for batch in loader:
        x, y, loss_mask = trainer.unpack_batch(batch)

        logits = model(x)
        loss = trainer.compute_loss(logits, y, loss_mask)

        trainer.optimizer.zero_grad()
        loss.backward()

        if config.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        trainer.optimizer.step()
        trainer.global_step += 1
```

Use this style for extra metrics, multiple losses, gradient accumulation,
custom schedulers, or experiment trackers.

## Checkpointing

For native ArcLM training, `Trainer.save()` writes a resumable checkpoint. Save
tokenizer metadata with the checkpoint so later training and inference can
restore the vocabulary correctly.

```python
trainer.save(
    config,
    vocab=tokenizer.vocab,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    tokenizer_metadata=tokenizer.to_checkpoint(),
)
```

Automatic checkpoint callbacks are available:

```python
from arclm import create_checkpoint_callback

checkpoint_callback = create_checkpoint_callback(
    config=config,
    tokenizer=tokenizer,
    vocab_size=tokenizer.get_vocab_size(),
)

trainer.train(
    loader,
    config.num_epochs,
    checkpoint_callback=checkpoint_callback,
    checkpoint_batch_interval=100,
)
```

The default callback saves to `config.model_path`. If you need separate
checkpoint folders, write your own callback or follow the pattern in the root
`train.py` script for Hugging Face models.

## Freezing Layers

Native `Trainer` can freeze or unfreeze parameters by name substring:

```python
trainer.freeze_layers("blocks")
trainer.freeze_layers("token_embedding")

print(trainer.get_frozen_layers_info())

trainer.unfreeze_layers("blocks")
trainer.unfreeze_layers()
```

This is useful for small native fine-tuning runs. Hugging Face LoRA SFT uses
PEFT through `train_sft(..., use_lora=True)` instead.

## Root `train.py`: Full Qwen Medical SFT Example

For a compact, runnable version of the same lower-level Hugging Face SFT
building blocks, see `examples/15_custom_hf_sft_loop.py`. It uses
`HuggingFaceSFTDataset`, `SFTDataCollator`, `load_sft_records`, `resolve_dtype`,
`model_input_device`, and optional `apply_lora` directly.

The repository root `train.py` is a complete workflow for:

- downloading `FreedomIntelligence/medical-o1-reasoning-SFT`;
- converting rows into ArcLM chat JSONL records;
- writing train/validation/test splits and metadata;
- analyzing token lengths with the Qwen tokenizer;
- loading `Qwen/Qwen3-0.6B`;
- optionally applying LoRA;
- training with gradient accumulation;
- saving periodic Hugging Face checkpoint folders;
- saving final adapter or full model output.

Install the extra dataset dependency before using it:

```bash
pip install -e .[peft]
pip install datasets
```

Prepare data only:

```bash
python train.py --prepare-only --max-samples 200
```

Run a short smoke training job:

```bash
python train.py --max-samples 200 --num-epochs 1 --max-steps 5 --checkpoint-steps 5
```

Run the default longer workflow:

```bash
python train.py
```

Useful options:

- `--model`: Hugging Face model ID, default `Qwen/Qwen3-0.6B`.
- `--output-dir`: final adapter or full model directory.
- `--no-lora`: save a full Hugging Face model instead of a LoRA adapter.
- `--checkpoint-steps`: save every N optimizer steps, or `0` to disable.
- `--max-samples`: limit dataset rows for debugging.
- `--skip-tokenizer-analysis`: skip token length analysis.
- `--gradient-checkpointing`: enable model gradient checkpointing when available.

## Root `test.py`: Load And Generate

The matching root `test.py` loads either:

- a LoRA adapter directory containing `adapter_config.json`; or
- a full Hugging Face model directory containing `config.json` and weights; or
- the latest loadable checkpoint under `MODEL_DIR/checkpoints`.

Run one prompt:

```bash
python test.py --model-dir models/qwen_medical_o1_lora --question "What is SFT?"
```

Use interactive mode:

```bash
python test.py --model-dir models/qwen_medical_o1_lora --interactive
```

Use a question file:

```bash
python test.py --model-dir models/qwen_medical_o1_lora --question-file data/question.txt
```

Common generation options:

- `--max-new-tokens`
- `--temperature`
- `--top-p`
- `--dtype`
- `--device-map`
- `--enable-thinking`

## What Makes A Good Custom Trainer?

Keep these rules in mind:

- Preserve the `[batch, time]` input and label shape unless you also replace
  the batch unpacking and loss code.
- Save tokenizer metadata with every native ArcLM checkpoint.
- Keep user/system prompt tokens masked out when doing instruction tuning.
- Reuse `InstructionDataset` for native assistant-only SFT instead of building
  masks by hand unless you need a different format.
- Use `train_sft()` for stable Hugging Face SFT. Use root `train.py` when you
  want the repository's full Qwen medical example with dataset preparation and
  periodic checkpoint folders.
- Run a tiny CPU smoke test before launching a long GPU job.
