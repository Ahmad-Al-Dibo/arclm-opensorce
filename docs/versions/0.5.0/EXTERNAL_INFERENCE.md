# External Inference Guide

The `external_inference` module provides a unified API for loading, inspecting, inferencing, training, and saving both native ArcLM checkpoints and external Hugging Face models with optional PEFT LoRA adapters.

## Overview

| Task | Function | Use Case |
| --- | --- | --- |
| Inspect a source | `inspect_model_source()` | Check model type before loading |
| Load any model | `load_any_model()` | Unified loading for ArcLM, HF, adapters |
| Single prediction | `predict_external()` | One-shot inference |
| Fine-tune external | `fine_tune_external_model()` | SFT on HF models |
| Train native | `train_native_model()` | Train ArcLM from scratch |
| Fine-tune native | `fine_tune_native_model()` | Fine-tune ArcLM checkpoint |
| Continue training | `continue_native_training()` | Resume native ArcLM training |
| Save model | `save_loaded_model()` | Export loaded model |

## Installation

The external inference features require optional dependencies:

```bash
pip install arclm[hf]  # For Hugging Face models
pip install arclm[peft]  # For LoRA adapters
```

## Inspect Models Before Loading

Use `inspect_model_source()` to understand what you're loading without full model loading overhead:

```python
from arclm import inspect_model_source

# Inspect a local ArcLM checkpoint
info = inspect_model_source("models/arclm_pretrained.pth")
print(info.format_report())
# Output:
# Source: models/arclm_pretrained.pth
# Source type: native_arclm
# Tokenizer files: found
# Weights: found
# Recommended loading strategy: load with arclm.load_model

# Inspect a Hugging Face model
info = inspect_model_source("meta-llama/Llama-2-7b-hf")
print(f"Type: {info.source_type}")  # hf_full_model

# Inspect a LoRA adapter
info = inspect_model_source("path/to/lora_adapter")
print(f"Base model: {info.base_model_name_or_path}")

# Check the detailed report
print(info.to_dict())  # Returns JSON-serializable dict
```

## Load Models for Inference

### Load a Native ArcLM Model

```python
from arclm import load_any_model

model = load_any_model("models/arclm_pretrained.pth")
print(model.generate("Hello, world!"))
```

### Load a Hugging Face Model

```python
from arclm import load_any_model

# Basic loading
model = load_any_model("meta-llama/Llama-2-7b-hf")

# With custom generation settings
model = load_any_model(
    "gpt2",
    max_new_tokens=256,
    temperature=0.8,
    top_p=0.95,
)

# With quantization
model = load_any_model(
    "meta-llama/Llama-2-13b-hf",
    load_in_4bit=True,  # or load_in_8bit=True
    device="cuda",
)

# With device mapping for multi-GPU
model = load_any_model(
    "mistralai/Mistral-7B",
    device_map="auto",  # Auto-distributes across available GPUs
)

# With custom dtype
model = load_any_model(
    "meta-llama/Llama-2-7b-hf",
    dtype="float16",  # or "bfloat16", "float32"
)
```

### Load a LoRA Adapter

```python
from arclm import load_any_model

# Load adapter with base model ID
model = load_any_model(
    "path/to/lora_adapter",
    base_model="meta-llama/Llama-2-7b-hf",
)

# Or if base_model is in adapter_config.json
model = load_any_model("path/to/lora_adapter")

# With custom tokenizer
model = load_any_model(
    "path/to/lora_adapter",
    base_model="meta-llama/Llama-2-7b-hf",
    tokenizer_path="path/to/custom_tokenizer",
)
```

## Generate Text

Once a model is loaded, use the unified inference API:

### Simple Generation

```python
model = load_any_model("gpt2")

# String prompt
text = model.generate("The meaning of life is")
print(text)

# Or use predict() as an alias
text = model.predict("The meaning of life is")
```

### Chat Messages

```python
model = load_any_model("meta-llama/Llama-2-7b-chat-hf")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is machine learning?"},
]
response = model.chat(messages)
print(response)
```

### With Custom Generation Settings

```python
model = load_any_model("gpt2")

# Override generation defaults
text = model.generate(
    "Explain quantum computing",
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.2,
)

# Batch prediction
prompts = [
    "Hello",
    "What is AI?",
    "Tell me a story",
]
results = model.batch_predict(prompts, temperature=0.8)
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

### Batch Processing

```python
model = load_any_model("gpt2")

prompts = ["Once upon a time", "In a galaxy far away", "The future is"]
responses = model.batch_predict(prompts, max_new_tokens=100)

for prompt, response in zip(prompts, responses):
    print(f"Input: {prompt}")
    print(f"Output: {response}")
    print("---")
```

## One-Shot Prediction

For simple load-and-predict workflows:

```python
from arclm import predict_external

# Load and predict in one call
response = predict_external(
    "gpt2",
    "The quick brown fox",
    max_new_tokens=50,
)
print(response)
```

## Web Inference UI

Run the Flask app to use external or native models from a browser:

```bash
pip install -e ".[web,hf]"
MODEL_SOURCE=gpt2 python app.py
```

`MODEL_SOURCE` can be a Hugging Face model ID, an `hf://owner/model` alias, a
downloaded Hugging Face model folder, a PEFT LoRA adapter folder, or a native
ArcLM checkpoint:

```bash
MODEL_SOURCE=hf://Qwen/Qwen2.5-0.5B-Instruct python app.py
MODEL_SOURCE=models/downloaded_hf_model python app.py
MODEL_SOURCE=models/my_lora_adapter BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct python app.py
MODEL_SOURCE=models/arclm.pth python app.py
```

Then open `http://localhost:5000`. The same server also exposes:

- `POST /model/inspect` to inspect a source without loading weights.
- `POST /model/load` to load or reload a selected model.
- `POST /generate` for full text generation.
- `POST /predict` for one-token-style clients.

## Move Model to Device

```python
import torch
from arclm import load_any_model

model = load_any_model("gpt2", device="cpu")

# Move to GPU
if torch.cuda.is_available():
    model.to("cuda:0")

# Generate on new device
text = model.generate("Hello")
```

## Save Loaded Models

Save loaded models with flexible export options:

```python
from arclm import load_any_model, ModelSaveConfig

# Load a LoRA adapter
model = load_any_model(
    "path/to/lora_adapter",
    base_model="meta-llama/Llama-2-7b-hf",
)

# Save as adapter only (default for LoRA)
config = ModelSaveConfig(
    save_mode="adapter_only",
    save_tokenizer=True,
    save_training_metadata=True,
)
model.save("output/lora_checkpoint", save_config=config)

# Or merge and save as full model
config = ModelSaveConfig(
    save_mode="merged_model",
    merge_lora=True,
)
model.save("output/merged_model", save_config=config)

# Or use save_pretrained() for HF compatibility
model.save_pretrained("output/hf_model")

# Or use export() for explicit workflows
model.export("output/exported")
```

### Save Configuration Options

```python
from arclm import ModelSaveConfig

config = ModelSaveConfig(
    save_enabled=True,              # Enable/disable saving
    save_mode="auto",               # auto|none|adapter_only|full_model|merged_model|native_arclm|all
    save_layout="model_id",         # flat|timestamped|checkpoint|model_id
    save_tokenizer=True,            # Save tokenizer files
    save_model_config=True,         # Save config.json
    save_generation_config=True,    # Save generation_config.json
    save_training_metadata=True,    # Save arclm_metadata.json
    save_adapter_config=True,       # Save adapter_config.json for LoRA
    save_processor=False,           # Save processor (vision models)
    save_readme=True,               # Save README with reload example
    save_safetensors=True,          # Use safetensors format
    merge_lora=False,               # Merge LoRA before saving
    overwrite=False,                # Overwrite existing output
)

model.save("output", save_config=config)
```

## Fine-Tune External Models

Fine-tune Hugging Face models using ArcLM's SFT backend:

```python
from arclm import fine_tune_external_model, ModelSaveConfig

result = fine_tune_external_model(
    model="gpt2",
    dataset="data/sft.jsonl",
    output_dir="models/gpt2_sft",
    method="sft",                   # Only method currently supported
    backend="huggingface",          # SFT backend
    use_lora=False,                 # Set True for LoRA adapters
    num_train_epochs=3,
    learning_rate=2e-5,
    batch_size=8,
    save_config=ModelSaveConfig(
        save_mode="full_model",
        save_tokenizer=True,
    ),
)

print(f"Model saved to: {result.output_dir}")
```

### Fine-Tune with LoRA

```python
result = fine_tune_external_model(
    model="meta-llama/Llama-2-7b-hf",
    dataset="data/sft.jsonl",
    output_dir="models/llama_lora",
    use_lora=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    num_train_epochs=2,
    learning_rate=5e-4,
)

print(f"LoRA adapter saved to: {result.output_dir}")
```

## Native ArcLM Training

### Train From Scratch

```python
from arclm import train_native_model

result = train_native_model(
    data="data/pretrain.txt",
    output="models/native_pretrained.pth",
    embed_dim=128,
    num_blocks=4,
    block_size=256,
    batch_size=16,
    num_epochs=5,
    learning_rate=3e-4,
)

print(f"Model saved to: {result.model_path}")
```

### Fine-Tune Native Checkpoint

```python
from arclm import fine_tune_native_model

result = fine_tune_native_model(
    checkpoint="models/native_pretrained.pth",
    data="data/finetune.txt",
    output="models/native_finetuned.pth",
    num_epochs=2,
    learning_rate=1e-4,
    freeze_backbone=True,
)
```

### Continue Native Training

```python
from arclm import continue_native_training

result = continue_native_training(
    checkpoint="models/native_finetuned.pth",
    data="data/more_data.txt",
    output="models/native_continued.pth",
    num_epochs=3,
    learning_rate=5e-5,
)
```

## Advanced Configuration

### Custom Generation Config

```python
from arclm import load_any_model, GenerationConfig

gen_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,
    pad_token_id=0,
    eos_token_id=2,
    stop=["Human:", "\n\n"],
    return_full_text=False,
)

model = load_any_model(
    "gpt2",
    generation_config=gen_config,
)

text = model.generate("Hello")
```

### External Model Config

```python
from arclm import load_any_model, ExternalModelConfig

config = ExternalModelConfig(
    source="meta-llama/Llama-2-7b-hf",
    base_model=None,
    device="cuda",
    device_map=None,
    dtype="float16",
    trust_remote_code=True,
    tokenizer_path=None,
    max_memory=None,
    load_in_8bit=False,
    load_in_4bit=False,
    enable_thinking=False,
    default_system_prompt="You are a helpful assistant.",
    fallback_to_simple_prompt=True,
    fallback_to_cpu=True,
    require_tokenizer=True,
    prefer_best_native=True,
)

model = load_any_model(**config.to_dict())
```

## Model Information Access

```python
model = load_any_model("gpt2")

# Access source information
print(f"Source type: {model.source_info.source_type}")
print(f"Tokenizer exists: {model.source_info.tokenizer_files_exist}")
print(f"Config: {model.source_info.config_json_exists}")

# Access model details
print(f"Device: {model.device}")
print(f"Tokenizer: {model.tokenizer}")
print(f"Config source: {model.source_info.resolved_source}")

# Access underlying model/tokenizer directly
print(f"Model type: {type(model.model)}")
print(f"Tokenizer type: {type(model.tokenizer)}")

# Use as context manager for resource cleanup
with load_any_model("gpt2") as m:
    output = m.generate("Hello")
```

## Error Handling

```python
from arclm import load_any_model, ModelSourceInfo

try:
    model = load_any_model("nonexistent/model")
except ValueError as e:
    print(f"Invalid model source: {e}")

# Inspect before loading to catch issues early
info = inspect_model_source("path/to/model")
if info.warnings:
    print(f"Warnings: {', '.join(info.warnings)}")

if info.source_type == "unknown":
    print("Unsupported model format")
else:
    model = load_any_model("path/to/model")
```

## Complete Workflow Example

```python
from arclm import (
    load_any_model,
    inspect_model_source,
    ModelSaveConfig,
    fine_tune_external_model,
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

## API Reference

### Key Classes

- **`ExternalModelConfig`**: Configuration for model loading
- **`ExternalLoadedModel`**: Loaded model wrapper with inference API
- **`GenerationConfig`**: Text generation defaults
- **`ModelSaveConfig`**: Save/export settings
- **`ModelSourceInfo`**: Model inspection result

### Key Functions

- **`inspect_model_source(source, **kwargs)`**: Inspect without loading
- **`load_any_model(source, **kwargs)`**: Load any supported model
- **`predict_external(source, prompt, **kwargs)`**: One-shot load and predict
- **`fine_tune_external_model(...)`**: Fine-tune HF models
- **`train_native_model(...)`**: Train native ArcLM
- **`fine_tune_native_model(...)`**: Fine-tune native checkpoint
- **`continue_native_training(...)`**: Continue native training
- **`save_loaded_model(loaded_model, output_dir, save_config)`**: Save loaded model

## See Also

- [USAGE.md](USAGE.md) - General ArcLM usage
- [FULL_FEATURE_GUIDE.md](FULL_FEATURE_GUIDE.md) - Complete framework overview
