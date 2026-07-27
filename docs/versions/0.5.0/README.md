# ArcLM

[![PyPI](https://img.shields.io/pypi/v/arclm.svg)](https://pypi.org/project/arclm/)
[![Python](https://img.shields.io/pypi/pyversions/arclm.svg)](https://pypi.org/project/arclm/)
[![License](https://img.shields.io/pypi/l/arclm.svg)](LICENSE)
[![Status](https://img.shields.io/badge/release-0.6.0-blue.svg)](docs/VERSIONING.md)

ArcLM is a compact open-source generative Ai framework for training native causal language models and running Hugging Face supervised fine-tuning workflows.

ArcLM is the canonical project name. Install and import it as `arclm`.

## Why ArcLM Exists

ArcLM is for developers who want a readable, local-first language-model stack instead of a black-box trainer. It keeps the important pieces visible: data loading, tokenization, dataset construction, model architecture, masking, optimization, checkpointing, inference, and diagnostics.

Use ArcLM when you want to:

- train small GPT-style models from text;
- fine-tune or continue compatible ArcLM checkpoints;
- run Hugging Face SFT with assistant-only labels and optional LoRA;
- load and generate from Hugging Face models and LoRA adapters;
- inspect and adapt Hugging Face checkpoints;
- inspect and adapt local checkpoints;
- prepare, clean, and format datasets before training;
- teach, debug, or extend the training loop without fighting a large framework.

## Key Features

| Area | What ArcLM 0.5.0 Provides |
| --- | --- |
| Native model | Compact GPT-style `ArcLM` / `MiniGPT` PyTorch model. |
| Training | `train_model()` for pretraining, next-token fine-tuning, and continued training. |
| SFT | `train_sft(backend="huggingface")` for Hugging Face causal LMs. |
| LoRA | Optional PEFT LoRA adapters for Hugging Face SFT. |
| Inference | `load_model()` and `predict()` for native checkpoints, Hugging Face models, and LoRA adapters. |
| Masking | Native instruction loss masks and HF `labels=-100` assistant-only labels. |
| Data | TXT, JSON, JSONL, CSV, cleaning, formatting, splitting, and preprocessing reports. |
| Tokenizers | Word tokenizer and SentencePiece tokenizer with checkpoint metadata. |
| Loading | Native checkpoints, raw PyTorch state dicts, safetensors, and Hugging Face sources. |
| Diagnostics | Perplexity, token accuracy, top-k predictions, coverage, and simple reports. |

## Installation

Default CPU-oriented install:

```bash
pip install "arclm[cpu]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

CUDA 12.1 install:

```bash
pip install "arclm[cuda]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

Plain PyPI install also works and installs ArcLM's core dependencies automatically, using your pip environment's configured PyTorch wheel source:

```bash
pip install arclm
```

Optional extras:

```bash
pip install "arclm[cpu]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple        # torch + torchvision + torchaudio from the selected CPU index
pip install "arclm[cuda]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple       # torch + torchvision + torchaudio from the selected CUDA index
pip install "arclm[cuda121]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple     # explicit alias for CUDA 12.1 PyTorch installs
pip install "arclm[preprocess]"  # HTML/YAML/progress helpers for preprocessing
pip install "arclm[peft]"        # PEFT LoRA support for train_sft(..., use_lora=True)
pip install "arclm[hf]"          # Hugging Face workflow helpers used by larger examples
```

Python package extras cannot embed pip's `--index-url`, so use the commands above when you need to force official CPU or CUDA PyTorch wheels.

## Supported Python Versions

ArcLM 0.5.0 supports Python `3.9`, `3.10`, `3.11`, and `3.12`.

Core runtime dependencies:

- `torch>=2.1,<3`
- `torchvision>=0.16,<1` and `torchaudio>=2.1,<3` when using the `cpu`, `cuda`, or `cuda121` extras
- `numpy>=1.24,<3`
- `sentencepiece>=0.2,<0.3`
- `transformers>=4.51,<6`

## Quick Start

This example creates a tiny text file, trains a native ArcLM checkpoint, and prints the saved path.

```python
from pathlib import Path

from arclm import train_model

Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("data/demo.txt").write_text(
    "ArcLM trains compact language models. "
    "Small examples make the pipeline easy to inspect. " * 20,
    encoding="utf-8",
)

result = train_model(
    mode="pretrain",
    data="data/demo.txt",
    output="models/demo_arclm.pth",
    tokenizer_type="word",
    max_vocab=200,
    embed_dim=32,
    num_blocks=1,
    block_size=16,
    batch_size=4,
    num_epochs=1,
    learning_rate=1e-3,
    validation_split=0.0,
    training_log_interval=0,
    device="cpu",
)

print(result.model_path)
```

## Core Concepts

| Concept | Meaning |
| --- | --- |
| Native checkpoint | A `.pth` file saved by ArcLM with model weights, config, optimizer state, tokenizer metadata, and history. |
| Pretraining | Train a new native `ArcLM` model from plain text using next-token prediction. |
| Fine-tuning | Adapt an existing checkpoint on formatted text with next-token loss. |
| Continued training | Resume or extend pretraining from a compatible native checkpoint and tokenizer. |
| SFT | Supervised fine-tuning on instruction/response or chat records. |
| Assistant-only loss | Train only on assistant answer tokens while using system/user tokens as context. |
| Loader | A utility that normalizes checkpoint sources before adaptation or inspection. |

## Usage Examples

### Train From Text

```python
from arclm import train_model

result = train_model(
    mode="pretrain",
    data="data/pretrain.txt",
    output="models/arclm_base.pth",
    tokenizer_type="word",
    embed_dim=64,
    num_blocks=2,
    block_size=64,
    batch_size=8,
    num_epochs=3,
)
```

### Continue Training

```python
from arclm import train_model

continued = train_model(
    mode="continue_training",
    checkpoint="models/arclm_base.pth",
    data="data/domain_text.txt",
    output="models/arclm_domain.pth",
    num_epochs=2,
    learning_rate=1e-4,
)
```

### Load For Inference

```python
from arclm import load_model

model = load_model("models/arclm_domain.pth", device="cpu")
print(model.predict("ArcLM is", max_new_tokens=40, top_p=0.9))
```

### Format Data

```python
from arclm import DataProcessor

dataset = (
    DataProcessor.load("data/qa.jsonl")
    .clean()
    .transform(
        format="instruction",
        mapping={"instruction": "question", "output": "answer"},
        template="Question: {instruction}\nAnswer: {output}",
    )
)

print(dataset.samples[0]["text"])
```

### Hugging Face SFT

Requires a compatible Hugging Face model and enough local compute for that model.

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

print(result.output_dir)
```

### Load, Predict, And Retrain A Hugging Face Model

Use `transformers` to load and generate from the base model, then use ArcLM to
fine-tune it with SFT. This example uses Qwen, but the same pattern works for
other causal language models that support `AutoModelForCausalLM`.

Install the optional training helpers first:

```bash
pip install "arclm[cpu,hf,peft]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# or, for CUDA 12.1:
pip install "arclm[cuda,hf,peft]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

```python
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arclm import train_sft

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Explain supervised fine-tuning in one sentence."


def render_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt


def load_hf_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt, max_new_tokens=80):
    text = render_prompt(tokenizer, prompt)
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


tokenizer, base_model = load_hf_model(MODEL_ID)
print("Before SFT:", generate(tokenizer, base_model, PROMPT))

data_path = Path("data/qwen_sft.jsonl")
data_path.parent.mkdir(parents=True, exist_ok=True)
records = [
    {
        "messages": [
            {"role": "user", "content": "What is ArcLM?"},
            {
                "role": "assistant",
                "content": "ArcLM is a compact Python toolkit for language-model training and SFT.",
            },
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "When should I use LoRA?"},
            {
                "role": "assistant",
                "content": "Use LoRA when you want to fine-tune a large model with fewer trainable parameters.",
            },
        ]
    },
]
data_path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")

result = train_sft(
    model=MODEL_ID,
    dataset=str(data_path),
    output_dir="models/qwen_lora_sft",
    backend="huggingface",
    assistant_only_loss=True,
    use_lora=True,
    batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    num_epochs=1,
    max_steps=5,
    max_length=512,
    device_map="auto",
)

print("Saved fine-tuned adapter:", result.adapter_path)

from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained(result.output_dir, trust_remote_code=True)
_, base_model = load_hf_model(MODEL_ID)
fine_tuned_model = PeftModel.from_pretrained(base_model, result.output_dir)
fine_tuned_model.eval()

print("After SFT:", generate(tokenizer, fine_tuned_model, PROMPT))
```

Set `use_lora=False` if you want ArcLM to save a full fine-tuned Hugging Face
model instead of a PEFT adapter.

### Native Instruction Tuning

```python
from torch.utils.data import DataLoader

from arclm import Config, InstructionDataset, Tokenizer, build_model, build_trainer

instructions = ["Explain SFT."]
responses = ["SFT trains a model on examples of desired responses."]
text = "\n".join(f"<|instruction|>\n{i}\n<|response|>\n{r}" for i, r in zip(instructions, responses))

tokenizer = Tokenizer(max_vocab=200, user_defined_symbols=["<|instruction|>", "<|response|>"])
tokenizer.build(text)

config = Config(
    vocab_size=tokenizer.get_vocab_size(),
    embed_dim=32,
    block_size=48,
    num_blocks=1,
    batch_size=1,
    num_epochs=1,
    model_path="models/native_sft.pth",
    device="cpu",
)

dataset = InstructionDataset(instructions, responses, tokenizer, config.block_size)
trainer = build_trainer(build_model(config), config)
trainer.train(DataLoader(dataset, batch_size=1), config.num_epochs)
```

## External Model Loading and Inference

ArcLM provides a unified API for loading, inspecting, generating with, fine-tuning, and saving native ArcLM checkpoints, Hugging Face models, and PEFT LoRA adapters.

### Inspect Models Before Loading

Use `inspect_model_source()` to understand a source without full model loading overhead:

```python
from arclm import inspect_model_source

# Inspect a local ArcLM checkpoint
info = inspect_model_source("models/arclm_pretrained.pth")
print(info.format_report())

# Inspect a Hugging Face model
info = inspect_model_source("meta-llama/Llama-2-7b-hf")
print(f"Type: {info.source_type}")  # hf_full_model

# Inspect a LoRA adapter
info = inspect_model_source("path/to/lora_adapter")
print(f"Base model: {info.base_model_name_or_path}")
```

### Load Models

Load any supported model with a unified API:

```python
from arclm import load_any_model

# Native ArcLM
model = load_any_model("models/arclm_pretrained.pth")

# Hugging Face full model
model = load_any_model("gpt2")

# Hugging Face with quantization
model = load_any_model(
    "meta-llama/Llama-2-7b-hf",
    load_in_4bit=True,
    device="cuda",
)

# LoRA adapter
model = load_any_model(
    "path/to/lora_adapter",
    base_model="meta-llama/Llama-2-7b-hf",
)
```

### Generate Text

```python
model = load_any_model("gpt2")

# Simple generation
text = model.generate("The meaning of life is")
print(text)

# Chat messages
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is AI?"},
]
response = model.chat(messages)

# Batch prediction
prompts = ["Hello", "What is AI?", "Tell me a story"]
results = model.batch_predict(prompts, temperature=0.8)

# Custom generation settings
text = model.generate(
    "Explain quantum computing",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2,
)
```

### One-Shot Prediction

For simple load-and-predict workflows:

```python
from arclm import predict_external

response = predict_external(
    "gpt2",
    "The quick brown fox",
    max_new_tokens=50,
)
```

### Save Loaded Models

```python
from arclm import load_any_model, ModelSaveConfig

model = load_any_model("gpt2")

# Save as full model
config = ModelSaveConfig(
    save_mode="full_model",
    save_tokenizer=True,
    save_training_metadata=True,
)
model.save("output/model", save_config=config)

# Or for LoRA adapters, save as adapter only
config = ModelSaveConfig(
    save_mode="adapter_only",
)
model.save("output/lora", save_config=config)

# Or merge and save
config = ModelSaveConfig(
    save_mode="merged_model",
    merge_lora=True,
)
model.save("output/merged", save_config=config)
```

### Fine-Tune External Models

Fine-tune Hugging Face models using ArcLM's SFT backend:

```python
from arclm import fine_tune_external_model

result = fine_tune_external_model(
    model="gpt2",
    dataset="data/sft.jsonl",
    output_dir="models/gpt2_sft",
    method="sft",
    backend="huggingface",
    use_lora=False,
    num_train_epochs=3,
    learning_rate=2e-5,
    batch_size=8,
)

print(f"Model saved to: {result.output_dir}")
```

With LoRA:

```python
result = fine_tune_external_model(
    model="meta-llama/Llama-2-7b-hf",
    dataset="data/sft.jsonl",
    output_dir="models/llama_lora",
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    num_train_epochs=2,
    learning_rate=5e-4,
)
```

### Advanced Configuration

```python
from arclm import (
    load_any_model, 
    GenerationConfig, 
    ExternalModelConfig,
    ModelSaveConfig
)

# Generation config
gen_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,
    stop=["Human:", "\n\n"],
)

model = load_any_model("gpt2", generation_config=gen_config)

# Or configure each part separately
config = ExternalModelConfig(
    source="meta-llama/Llama-2-7b-hf",
    device="cuda",
    dtype="float16",
    load_in_4bit=False,
    default_system_prompt="You are helpful.",
)

model = load_any_model(**config.to_dict())

# Save config
save_config = ModelSaveConfig(
    save_enabled=True,
    save_mode="auto",
    save_tokenizer=True,
    save_training_metadata=True,
    save_safetensors=True,
    overwrite=False,
)

model.save("output", save_config=save_config)
```

### Complete Workflow

```python
from arclm import (
    load_any_model,
    inspect_model_source,
    fine_tune_external_model,
    ModelSaveConfig,
)

# 1. Inspect the source
print("Inspecting model...")
info = inspect_model_source("meta-llama/Llama-2-7b-hf")
print(info.format_report())

# 2. Load for inference
print("\nLoading model...")
model = load_any_model(
    "meta-llama/Llama-2-7b-hf",
    device="cuda",
    dtype="float16",
)

# 3. Test inference
print("\nTesting inference...")
response = model.generate("What is machine learning?", max_new_tokens=100)
print(f"Response: {response}")

# 4. Fine-tune
print("\nFine-tuning...")
result = fine_tune_external_model(
    model="meta-llama/Llama-2-7b-hf",
    dataset="data/training.jsonl",
    output_dir="models/llama_ft",
    use_lora=True,
    num_train_epochs=1,
)

# 5. Save with custom config
print("\nSaving model...")
save_config = ModelSaveConfig(
    save_mode="merged_model",
    save_tokenizer=True,
    save_training_metadata=True,
)
model.save("models/final", save_config=save_config)

print("Done!")
```

## Provider Support

ArcLM 0.5.0 is a training and checkpoint toolkit, not a hosted LLM API client.

| Source / Provider | Status |
| --- | --- |
| Native ArcLM checkpoints | Supported for training, continued training, loading, and inference. |
| Hugging Face causal LMs | Supported for SFT through `train_sft(backend="huggingface")`. |
| Hugging Face folders/model IDs | Supported by external loaders where `transformers` can load the source. |
| Raw PyTorch checkpoints | Supported when tensor shapes are compatible with ArcLM adaptation. |
| safetensors files | Supported when `safetensors` is installed. |
| OpenAI / Anthropic / Google Gemini / Ollama APIs | Not implemented in 0.5.0. Use their official SDKs beside ArcLM if you need hosted inference. |

## Configuration

Use `Config` directly or `create_config()` when you want unknown-key validation.

```python
from arclm import create_config

config = create_config(
    tokenizer_type="sentencepiece",
    max_vocab=8000,
    embed_dim=128,
    num_blocks=4,
    block_size=128,
    batch_size=8,
    learning_rate=3e-4,
    device="cpu",
)
```

Common fields:

| Field | Purpose |
| --- | --- |
| `data_path`, `model_path` | Input data and output checkpoint paths. |
| `tokenizer_type`, `max_vocab` | `word` or `sentencepiece` tokenizer settings. |
| `embed_dim`, `num_blocks`, `block_size`, `dropout` | Native model shape. |
| `batch_size`, `num_epochs`, `learning_rate`, `weight_decay` | Training settings. |
| `validation_split`, `early_stopping_patience` | Evaluation and stopping behavior. |
| `freeze_backbone`, `freeze_embedding` | Native fine-tuning controls. |
| `checkpoint_interval`, `checkpoint_batch_interval` | Native checkpoint frequency. |

## Project Structure

```text
arclm/
  config.py              # Config and presets
  data.py                # token loading, splitting, DataBundle
  data_processor.py      # JSON/JSONL/CSV/TXT formatting helpers
  dataset.py             # next-token dataset and DataLoader helper
  model.py               # native ArcLM transformer
  trainer.py             # native training loop
  pipeline.py            # train_model and checkpoint helpers
  sft.py                 # Hugging Face SFT backend
  inference.py           # load_model and predict
  loaders/               # checkpoint/source loaders
  preprocess/            # JSONL cleaning and reports
  tokenizers/            # tokenizer exports
  training/              # extension classes
docs/
examples/
tests/
```

## Documentation

- [Usage Guide](docs/USAGE.md)
- [Full Feature Guide](docs/FULL_FEATURE_GUIDE.md)
- [API Reference](docs/reference/LIBRARY_README.md)
- [Custom Trainers](docs/CUSTOM_TRAINER.md)
- [Release Guide](docs/VERSIONING.md)
- [Changelog](CHANGELOG.md)

## Examples Directory

The numbered examples are short, independent scripts focused on one concept each:

```text
examples/
  01_quickstart.py
  02_tokenization.py
  03_data_processing.py
  04_pretraining.py
  05_continue_training.py
  06_finetuning.py
  07_native_sft.py
  08_huggingface_sft.py
  09_lora_sft.py
  10_preprocess_pipeline.py
  11_inference.py
  12_smart_loader.py
  13_diagnostics.py
  14_custom_trainer.py
  15_custom_hf_sft_loop.py
```

Advanced workflow:

- [examples/15_custom_hf_sft_loop.py](examples/15_custom_hf_sft_loop.py) demonstrates a complete custom Hugging Face SFT loop using ArcLM's public SFT helper classes.
- [examples/qwen3_0_6b_sft/](examples/qwen3_0_6b_sft/) demonstrates a fuller Qwen SFT loop.

## API Overview

| API | Use |
| --- | --- |
| `train_model` | High-level native pretraining, fine-tuning, and continued training. |
| `TrainingResult` | Return object from `train_model`. |
| `train_sft` | Hugging Face causal-LM SFT with optional PEFT LoRA. |
| `SFTTrainingResult` | Return object from `train_sft`. |
| `arclm.sft.HuggingFaceSFTDataset`, `SFTDataCollator` | Advanced Hugging Face SFT building blocks for custom loops. |
| `load_model`, `predict` | Native checkpoint inference. |
| `Config`, `create_config` | Training and model configuration. |
| `ArcLM`, `MiniGPT` | Native causal language model. |
| `Tokenizer`, `SentencePieceTokenizer` | Native tokenizers. |
| `DataProcessor` | Structured dataset loading, cleaning, formatting, and splitting. |
| `PreprocessPipeline` | JSONL preprocessing with reports. |
| `InstructionDataset` | Native assistant-only masked SFT dataset. |
| `Trainer` | Native training loop and extension point. |
| `load_external_model`, `SmartLoader` | External source loading and inspection. |
| `calculate_metrics`, `predict_top_k` | Diagnostics and evaluation helpers. |

## FAQ

**Is ArcLM a replacement for Transformers?**

No. ArcLM uses `transformers` for Hugging Face SFT and provides its own small native model/training path for inspectable local experiments.

**Can I train large production LLMs with ArcLM?**

ArcLM 0.5.0 is best for compact models, teaching, prototyping, and focused SFT workflows. It is not a distributed training framework.

**Does ArcLM support OpenAI, Anthropic, Gemini, or Ollama providers?**

No hosted-provider clients are implemented in 0.5.0.

**Why does native SFT require `InstructionDataset` and `Trainer`?**

The high-level `train_sft()` API currently targets Hugging Face models. Native assistant-only SFT is available through lower-level building blocks.

**Where do checkpoints store tokenizer data?**

`Trainer.save()` and `train_model()` save tokenizer metadata alongside weights so `load_model()` and continued training can restore the same token mapping.

## Performance Notes

- CPU training is useful for smoke tests and small examples, but GPU is strongly recommended for real runs.
- Increase `block_size`, `embed_dim`, `num_blocks`, and vocabulary size gradually; each one increases memory use.
- Use `use_lora=True` for larger Hugging Face SFT runs when full fine-tuning is too expensive.
- Keep `validation_split` small but non-zero when comparing experiments.
- Use tiny datasets and `max_steps` for SFT smoke tests before launching long jobs.

## License

ArcLM is licensed under the [Apache License 2.0](LICENSE).
