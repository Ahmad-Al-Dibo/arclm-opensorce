# ArcLM Usage Guide

This guide shows the supported ArcLM 0.5.0 workflows. For a shorter overview, start with the root `README.md`.

## Install

Default CPU-oriented install:

```bash
pip install "arclm[cpu]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

CUDA 12.1 install:

```bash
pip install "arclm[cuda]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

Plain PyPI install also works and installs the core dependencies from your configured package indexes:

```bash
pip install arclm
```

For development:

```bash
pip install -e .[dev]
```

Optional extras:

```bash
pip install -e .[cpu]
pip install -e .[cuda]
pip install -e .[preprocess]
pip install -e .[peft]
pip install -e .[hf]
```

Extras cannot carry pip index flags. Use the PyTorch CPU or CUDA index commands above when you need a specific PyTorch wheel flavor.

## Workflow Map

| Goal | Recommended API |
| --- | --- |
| Train a small native model | `train_model(mode="pretrain", ...)` |
| Fine-tune a native/adapted checkpoint | `train_model(mode="finetune", ...)` |
| Continue compatible native training | `train_model(mode="continue_training", ...)` |
| Run Hugging Face SFT | `train_sft(backend="huggingface", ...)` |
| Train native assistant-only examples | `InstructionDataset` + `Trainer` |
| Load native checkpoints | `load_model(...)` or `predict(...)` |
| Prepare records | `DataProcessor` |
| Clean JSONL datasets | `arclm.preprocess.PreprocessPipeline` |

## Train From Text

```python
from arclm import train_model

result = train_model(
    mode="pretrain",
    data="data/pretrain.txt",
    output="models/arclm_base.pth",
    tokenizer_type="word",
    max_vocab=2000,
    embed_dim=64,
    num_blocks=2,
    block_size=64,
    batch_size=8,
    num_epochs=3,
    learning_rate=3e-4,
    validation_split=0.1,
    device="cpu",
)

print(result.model_path)
```

`train_model()` prepares data, builds/reuses a tokenizer, creates dataloaders, builds the native model, trains, writes metrics, and saves a checkpoint.

## Fine-tune A Checkpoint

```python
from arclm import train_model

result = train_model(
    mode="finetune",
    checkpoint="models/arclm_base.pth",
    data="data/finetune.txt",
    output="models/arclm_finetuned.pth",
    num_epochs=2,
    learning_rate=2e-5,
    freeze_backbone=True,
    use_discriminative_lr=True,
)
```

Fine-tuning uses next-token loss over the formatted text file. It is different from assistant-only SFT.

## Continue Training

```python
from arclm import train_model

result = train_model(
    mode="continue_training",
    checkpoint="models/arclm_base.pth",
    data="data/domain_text.txt",
    output="models/arclm_continued.pth",
    num_epochs=2,
    learning_rate=1e-4,
)
```

Continued training requires a compatible native checkpoint and tokenizer metadata.

## Hugging Face SFT

Use `train_sft()` for Hugging Face causal language models.

```python
from arclm import train_sft

result = train_sft(
    model="Qwen/Qwen3-0.6B",
    dataset="data/sft.jsonl",
    output_dir="models/qwen_sft",
    backend="huggingface",
    assistant_only_loss=True,
    use_lora=True,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_epochs=1,
    max_length=1024,
    device_map="auto",
)
```

Supported SFT record shapes:

```json
{"messages":[{"role":"user","content":"What is SFT?"},{"role":"assistant","content":"SFT trains on desired responses."}]}
```

```json
{"instruction":"Explain SFT.","output":"SFT teaches a model to follow instructions."}
```

```json
{"conversations":[{"from":"human","value":"What is ArcLM?"},{"from":"gpt","value":"A compact LM toolkit."}]}
```

When `assistant_only_loss=True`, system/user tokens remain context and assistant answer tokens receive labels.

## Native Assistant-only SFT

Native ArcLM assistant-only training uses `InstructionDataset` and `Trainer`.

```python
from torch.utils.data import DataLoader

from arclm import Config, InstructionDataset, Tokenizer, build_model, build_trainer

instructions = ["Explain assistant-only loss."]
responses = ["It computes loss only on assistant response tokens."]
text = "\n".join(f"<|instruction|>\n{i}\n<|response|>\n{r}" for i, r in zip(instructions, responses))

tokenizer = Tokenizer(max_vocab=500, user_defined_symbols=["<|instruction|>", "<|response|>"])
tokenizer.build(text)

config = Config(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=64,
    block_size=64,
    num_blocks=2,
    batch_size=1,
    num_epochs=1,
    model_path="models/native_sft.pth",
    device="cpu",
)

dataset = InstructionDataset(instructions, responses, tokenizer, config.block_size)
trainer = build_trainer(build_model(config), config)
trainer.train(DataLoader(dataset, batch_size=1), config.num_epochs)
```

## DataProcessor

Use `DataProcessor` for lightweight record loading and formatting.

```python
from arclm import DataProcessor

processed = (
    DataProcessor.load("data/qa.jsonl")
    .clean()
    .filter(lambda row: bool(row.get("question")))
    .transform(
        format="instruction",
        mapping={"instruction": "question", "output": "answer"},
        template="Question: {instruction}\nAnswer: {output}",
    )
)
```

Write formatted text for next-token fine-tuning:

```python
from pathlib import Path

Path("data/finetune.txt").write_text(
    "\n\n".join(sample["text"] for sample in processed.samples),
    encoding="utf-8",
)
```

## Preprocessing Pipeline

`arclm.preprocess` is JSONL-first and can normalize text, redact PII, filter low-quality rows, deduplicate, and write reports.

```python
from arclm.preprocess import PreprocessConfig, PreprocessPipeline

report = PreprocessPipeline(
    PreprocessConfig(
        text_field="text",
        output_field="text",
        min_chars=20,
        min_words=4,
        drop_emails=True,
        redact_pii=True,
    )
).run("data/raw.jsonl", "data/cleaned.jsonl", "reports/preprocess")

print(report["written"])
```

## Load And Generate
SAfrom arclm import load_any_model, inspect_model_source

# Inspect before loading
info = inspect_model_source("models/arclm_base.pth")
print(info.format_report())

# Load and generate
model = load_any_model("gpt2")
print(model.generate("Hello world"))- `generator`
- `config`
- `device`
- `predict(...)`

## External Source Loading

Use the unified external inference API to load, inspect, and generate with any supported model:

```python
from arclm import load_any_model, inspect_model_source

# Inspect before loading
info = inspect_model_source("models/arclm_base.pth")
print(info.format_report())

# Load and generate
model = load_any_model("gpt2")
print(model.generate("Hello world"))
```

Supported sources include native ArcLM checkpoints, Hugging Face causal LMs, PEFT LoRA adapters, raw PyTorch state dicts, and safetensors files. See the main README for comprehensive external inference examples.

## Examples

Run a focused example:

```bash
python examples/01_quickstart.py
python examples/03_data_processing.py
python examples/10_preprocess_pipeline.py
```

Hugging Face examples may download models and need more memory:

```bash
python examples/08_huggingface_sft.py
python examples/09_lora_sft.py
```

## Troubleshooting

`Tokenizer vocabulary size is incompatible with checkpoint`

Reuse the checkpoint tokenizer or train a new model with a matching vocabulary.

`Masked SFT loss is zero`

Make sure responses are non-empty and `block_size` or `max_length` is long enough to include assistant tokens.

`Missing PEFT`

Install `pip install -e .[peft]` or set `use_lora=False`.

`CUDA out of memory`

Use LoRA, reduce `batch_size`, reduce `max_length`, increase gradient accumulation, or choose a smaller model.

`Provider SDK support`

ArcLM 0.5.0 does not include hosted provider clients. Use the official provider SDK beside ArcLM when you need hosted inference.
