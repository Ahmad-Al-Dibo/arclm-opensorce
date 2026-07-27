# ArcLM Full Feature Guide

This guide explains the current ArcLM framework surface. It is longer than
`USAGE.md` and is meant to help contributors and users understand how the pieces
fit together.

## What ArcLM Is

ArcLM is a compact PyTorch language-model framework for:

- training small native ArcLM causal language models;
- fine-tuning or continuing compatible ArcLM checkpoints;
- supervised fine-tuning Hugging Face causal LMs through `train_sft()`;
- optional PEFT LoRA adapters for Hugging Face SFT;
- assistant-only loss masking for instruction tuning;
- dataset loading, formatting, preprocessing, diagnostics, and tracking;
- external checkpoint inspection and adaptation.

ArcLM is not a distributed training framework and is not a hosted provider
client. It does not implement OpenAI, Anthropic, Google Gemini, or Ollama API
clients; DPO/RLHF/PPO; reward modeling; native ArcLM LoRA layers; or multi-node
training.

## Install

```bash
pip install "arclm[cpu]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install "arclm[cuda]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

For development from a local checkout:

```bash
pip install -e .[dev]
pip install -e .[preprocess]
pip install -e .[peft]
```

Runtime dependencies are `torch`, `numpy`, `sentencepiece`, and
`transformers>=4.51`. The `cpu`, `cuda`, and `cuda121` extras also request
`torchvision` and `torchaudio`; the PyTorch index URL selects the CPU or CUDA
wheel flavor. The `peft` extra is only needed for `train_sft(..., use_lora=True)`.

## Architecture Overview

Main packages:

- `arclm.config`: `Config`, `create_config`, and tuning presets.
- `arclm.tokenizer` / `arclm.tokenizers`: word and SentencePiece tokenizers.
- `arclm.data`: token loading, splitting, and `prepare_data`.
- `arclm.dataset`: sliding-window `TextDataset` and DataLoader helper.
- `arclm.model`: native `ArcLM` / `MiniGPT` model.
- `arclm.trainer`: native training loop with tuple and masked dict batches.
- `arclm.pipeline`: `train_model`, checkpoint helpers, model/trainer builders.
- `arclm.sft`: Hugging Face SFT backend and `train_sft`.
- `arclm.loaders`: external source loaders and SmartLoader.
- `arclm.inference` / `arclm.generator`: checkpoint loading and generation.
- `arclm.preprocess`: JSONL cleaning, redaction, deduplication, reports.
- `arclm.diagnostics`: metrics, top-k predictions, coverage, benchmarks.
- `arclm.tracking`: local experiment logging.
- `arclm.training`: extension base classes and `UnifiedPipeline`.

## Training Modes

| Mode | API | Data | Checkpoint |
| --- | --- | --- | --- |
| Pretraining | `train_model(mode="pretrain")` | plain text | native ArcLM |
| Next-token fine-tuning | `train_model(mode="finetune")` | formatted text | native ArcLM/adapted source |
| Continued training | `train_model(mode="continue_training")` | plain text | compatible native ArcLM |
| Hugging Face SFT | `train_sft(backend="huggingface")` | structured SFT JSON/JSONL/CSV | HF model or LoRA adapter |
| Native ArcLM instruction tuning | `InstructionDataset` + `Trainer` | instruction/response lists | native ArcLM |
| Custom native training | `Trainer` subclass or manual loop | tuple or dict batches | native ArcLM |

Preference training is not implemented.

## Native Pretraining

```python
from arclm import train_model

result = train_model(
    mode="pretrain",
    data="data/data.txt",
    output="models/arclm_pretrained.pth",
    tokenizer_type="sentencepiece",
    max_vocab=8000,
    embed_dim=128,
    num_blocks=4,
    block_size=128,
    batch_size=16,
    num_epochs=5,
    learning_rate=3e-4,
    validation_split=0.1,
)
```

The result contains `mode`, `model_path`, `config`, `history`, `vocab_size`,
`tokenizer`, and optional `checkpoint_source`.

## Native Fine-tuning

```python
from arclm import train_model

result = train_model(
    mode="finetune",
    checkpoint="models/arclm_pretrained.pth",
    data="data/finetune.txt",
    output="models/arclm_finetuned.pth",
    num_epochs=2,
    learning_rate=2e-5,
    freeze_backbone=True,
    use_discriminative_lr=True,
)
```

This uses next-token loss over all labels. It is useful when your file already
contains the exact prompt/answer text you want the native model to learn.

## Continued Training

```python
continued = train_model(
    mode="continue_training",
    checkpoint="models/arclm_pretrained.pth",
    data="data/more_pretraining.txt",
    output="models/arclm_continued.pth",
    num_epochs=5,
)
```

Continued training requires a compatible tokenizer and model shape.

## Hugging Face SFT And LoRA

`train_sft()` is the public ArcLM SFT API for Hugging Face causal LMs.

```python
from arclm import train_sft

result = train_sft(
    model="Qwen/Qwen3-0.6B",
    dataset="examples/qwen3_0_6b_sft/data/sample_sft.jsonl",
    output_dir="examples/qwen3_0_6b_sft/output/qwen3_0_6b_sft_lora",
    backend="huggingface",
    assistant_only_loss=True,
    use_lora=True,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_epochs=1,
    max_length=1024,
    dtype="auto",
    device_map="auto",
)
```

Important behavior:

- `backend="huggingface"` is the only implemented `train_sft` backend.
- `assistant_only_loss=True` ignores system/user labels and trains only on assistant answer tokens.
- `use_lora=True` requires PEFT and saves a LoRA adapter.
- `use_lora=False` saves a full Hugging Face model.
- Qwen3 chat templates are supported; the example tries `enable_thinking=False`.

Complete workflow:

```text
examples/qwen3_0_6b_sft/README.md
```

## Native Assistant-only SFT

For native ArcLM checkpoints:

```python
from torch.utils.data import DataLoader
from arclm import InstructionDataset, build_trainer

dataset = InstructionDataset(
    instructions=["What is ArcLM?"],
    responses=["ArcLM is a compact PyTorch language-model framework."],
    tokenizer=tokenizer,
    block_size=128,
)

trainer = build_trainer(model, config)
trainer.train(DataLoader(dataset, batch_size=1), epochs=1)
```

`InstructionDataset` returns:

- `x`: input IDs
- `y`: next-token labels
- `mask`: label positions that contribute to loss

The mask is shifted to label positions so assistant response labels are active.

## Custom Native Trainers

The native `Trainer` is the main extension point when the high-level
`train_model()` helper is too fixed. It supports:

- tuple batches from `TextDataset`: `(x, y)`;
- dict batches from instruction datasets: `{"x": x, "y": y, "mask": mask}`;
- validation and early stopping;
- gradient clipping through `config.grad_clip`;
- checkpoint callbacks by epoch or batch interval;
- layer freezing through `freeze_layers()` and `unfreeze_layers()`;
- save/load of resumable native ArcLM checkpoints.

The batch contract is simple: `x` and `y` are `[batch, time]` tensors, native
`ArcLM.forward(x)` returns `[batch, time, vocab]` logits, and an optional
`mask` selects which label positions contribute to loss.

Subclass `Trainer` when you want a different loss:

```python
import torch.nn.functional as F
from arclm import Trainer


class LabelSmoothingTrainer(Trainer):
    def compute_loss(self, logits, y, loss_mask=None):
        batch_size, steps, vocab_size = logits.shape
        token_loss = F.cross_entropy(
            logits.reshape(batch_size * steps, vocab_size),
            y.reshape(batch_size * steps),
            reduction="none",
            label_smoothing=0.05,
        ).reshape(batch_size, steps)

        if loss_mask is None:
            return token_loss.mean()

        loss_mask = loss_mask.to(token_loss.device, dtype=token_loss.dtype)
        return (token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
```

Write a manual loop when you need custom schedulers, gradient accumulation,
multiple losses, extra logging, or integration with another experiment system:

```python
for batch in loader:
    x, y, loss_mask = trainer.unpack_batch(batch)
    logits = trainer.model(x)
    loss = trainer.compute_loss(logits, y, loss_mask)

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()
```

Save tokenizer metadata with native checkpoints:

```python
trainer.save(
    config,
    vocab=tokenizer.vocab,
    stoi=tokenizer.stoi,
    itos=tokenizer.itos,
    tokenizer_metadata=tokenizer.to_checkpoint(),
)
```

See `CUSTOM_TRAINER.md` for a complete low-level guide.

## Masks

ArcLM has three mask concepts:

- Causal mask: internal to native `SelfAttention`, prevents future-token attention.
- Loss mask: `InstructionDataset`/`Trainer` and `train_sft` label construction.
- Attention mask: used by Hugging Face `train_sft`; not an input to native `ArcLM.forward()` yet.

## Data Formats

Plain text for next-token training:

```text
ArcLM trains causal language models.
```

OpenAI-style SFT:

```json
{"messages":[{"role":"system","content":"Be helpful."},{"role":"user","content":"What is SFT?"},{"role":"assistant","content":"SFT trains on instruction-response examples."}]}
```

Instruction/output SFT:

```json
{"instruction":"Summarize ArcLM.","output":"ArcLM is a compact local language-model framework."}
```

ShareGPT-style:

```json
{"conversations":[{"from":"human","value":"What is ArcLM?"},{"from":"gpt","value":"A compact LM framework."}]}
```

## Tokenizers

Native ArcLM supports:

- `Tokenizer`: simple word-level tokenizer.
- `SentencePieceTokenizer`: subword tokenizer with BPE or unigram model types.
- `TokenizerFactory`: register and create tokenizers by name.

Use `encode_text()` for raw strings and `encode()` only for pre-tokenized token
lists. Save tokenizer metadata with native checkpoints.

## Checkpoints

Native `Trainer.save()` writes:

- `model_state_dict`
- `optimizer_state_dict`
- config
- vocab/stoi/itos
- tokenizer metadata
- block size and vocab size
- current epoch, batch, global step
- train history and best validation state

Restore a tokenizer:

```python
from arclm import tokenizer_from_checkpoint

tokenizer = tokenizer_from_checkpoint("models/arclm_pretrained.pth")
```

## Inference

```python
from arclm import load_model

loaded = load_model("models/arclm_finetuned.pth")
print(loaded.predict("ArcLM is", max_new_tokens=50, top_p=0.9))
```

`LoadedModel` provides `model`, `generator`, `config`, `device`, and
`predict(...)`.

## External Loading and Inference

The unified external inference API provides a high-level interface for loading, inspecting, generating with, fine-tuning, and saving any supported model:

- native ArcLM checkpoints;
- Hugging Face causal language models;
- PEFT LoRA adapters;
- raw PyTorch state dicts;
- `.safetensors` files.

Use `inspect_model_source()` to understand a source before loading:

```python
from arclm import inspect_model_source, load_any_model

info = inspect_model_source("meta-llama/Llama-2-7b-hf")
print(info.format_report())

model = load_any_model(
    "meta-llama/Llama-2-7b-hf",
    device="cuda",
    dtype="float16",
)
print(model.generate("Hello"))
```

Comprehensive external inference examples are available in the main README.

## Preprocessing

`arclm.preprocess` can:

- strip HTML;
- normalize text;
- redact emails, phones, IPs;
- filter low-quality rows;
- exact and near deduplicate;
- write JSON and HTML reports.

```python
from arclm.preprocess import PreprocessConfig, PreprocessPipeline

report = PreprocessPipeline(PreprocessConfig()).run(
    "data/raw.jsonl",
    "data/cleaned.jsonl",
    "reports/preprocess",
)
```

## Diagnostics

Useful helpers:

- `predict_top_k`
- `format_top_k_predictions`
- `calculate_metrics`
- `calculate_perplexity`
- `export_metrics_to_json`
- `format_tokenizer_coverage_report`
- `run_long_context_evaluation`

```python
from arclm import predict_top_k, format_top_k_predictions

predictions = predict_top_k(
    loaded.model,
    loaded.generator.stoi,
    loaded.generator.itos,
    loaded.config.block_size,
    loaded.device,
    "machine learning",
    k=5,
    tokenizer=loaded.generator.tokenizer,
)
print(format_top_k_predictions("machine learning", predictions))
```

## Tracking

Local experiment tracking:

```python
from arclm.tracking import create_experiment

tracker = create_experiment("baseline", experiment_dir="experiments")
tracker.log_config(config)
tracker.log_metrics({"loss": 2.5}, step=1)
tracker.end()
```

Optional MLflow and Weights & Biases backends are used only if installed.

## Flask API

`app.py` exposes:

- `GET /health`
- `POST /predict`
- `POST /generate`

Run:

```bash
set MODEL_PATH=models/arclm_finetuned.pth
python app.py
```

## Example Scripts

The numbered examples are the maintained public examples:

- `examples/01_quickstart.py`: tiny native training run.
- `examples/02_tokenization.py`: word tokenizer basics.
- `examples/03_data_processing.py`: structured data formatting.
- `examples/04_pretraining.py`: native pretraining.
- `examples/05_continue_training.py`: compatible continuation.
- `examples/06_finetuning.py`: next-token fine-tuning.
- `examples/07_native_sft.py`: native masked instruction tuning.
- `examples/08_huggingface_sft.py`: Hugging Face SFT smoke workflow.
- `examples/09_lora_sft.py`: Hugging Face PEFT LoRA setup.
- `examples/10_preprocess_pipeline.py`: JSONL preprocessing reports.
- `examples/11_inference.py`: load and generate from a native checkpoint.
- `examples/12_smart_loader.py`: source inspection.
- `examples/13_diagnostics.py`: lightweight diagnostics.
- `examples/14_custom_trainer.py`: custom loss through `Trainer`.
- `examples/15_custom_hf_sft_loop.py`: custom Hugging Face SFT loop using ArcLM's public SFT helper classes.

Advanced workflows:

- `examples/qwen3_0_6b_sft/`: Qwen3 base/SFT/load/benchmark workflow.
- root `train.py`: full Qwen medical SFT workflow with dataset conversion, token analysis, LoRA/full-model training, and periodic Hugging Face checkpoints.
- root `test.py`: matching Qwen adapter/full-model loader and generation script.

## Troubleshooting

`Special tokens become <UNK>`

Use the same `user_defined_symbols` when training and fine-tuning. Use
`encode_text()` for raw strings.

`Tokenizer is incompatible`

Reuse the checkpoint tokenizer or train a new compatible model head.

`Missing PEFT`

Install `pip install -e .[peft]`, or call `train_sft(..., use_lora=False)`.

`Qwen training is slow on CPU`

Qwen3-0.6B can load on CPU but training is slow. Use a CUDA GPU for the full
example, or test the API with a smaller Hugging Face model.

`Which interface should I use?`

Use the Python APIs and numbered examples as the canonical 0.5.0 workflows.
