# ArcLM

ArcLM is a focused Python framework for preparing language-model data and building reproducible workflows for causal language models.

ArcLM `0.8.0.dev0` is not a general machine-learning framework and should not be treated as production-ready yet. Its strongest supported paths are data-first workflows, streaming dataset preparation, native ArcLM causal-language-model checkpoints, and certified tiny GPT-2 Hugging Face integration tests.

## Purpose

ArcLM helps developers move through the practical language-model workflow:

```text
Raw data -> Loading -> Cleaning -> Validation -> Transformation -> Formatting
-> Tokenization -> Model loading -> Training or fine-tuning -> Evaluation
-> Inference -> Reporting
```

The framework puts dataset preparation first because most training and fine-tuning failures start before the model is loaded: inconsistent records, missing fields, duplicated samples, tokenizer mismatches, and undocumented formatting choices.

## Main Features

- Load JSON, JSONL, CSV, TXT, or custom in-memory datasets with `DataProcessor`.
- Clean, filter, transform, split, and tokenize records with composable dataset helpers.
- Run JSONL preprocessing reports with `PreprocessPipeline`.
- Build word or SentencePiece tokenizers with `Tokenizer` and `SentencePieceTokenizer`.
- Train compact native decoder-only ArcLM models with `train_model`.
- Load native checkpoints with `load_model`.
- Inspect and load Hugging Face causal-LM sources with `inspect_model_source` and `load_any_model`.
- Run Hugging Face SFT with `train_sft` when optional dependencies and hardware are available.
- Generate metrics and diagnostics for native ArcLM checkpoints.
- Start the optional Flask simple interface with `python -m arclm --run simple-interface`.

## Project Status

Development version: `0.8.0.dev0`. Current released version: `0.6.1`.

Status: pre-1.0 framework development. Public APIs are usable but still need stronger validation, automated documentation checks, model-family verification, and a formal deprecation policy before a stable production release.

## Supported Model Focus

ArcLM initially focuses on:

- Causal language models
- Decoder-only transformer models
- Models compatible with causal language modeling workflows

Official support currently means the model path has verified loading, tokenizer loading, causal-LM behavior, inference or training where claimed, documented limitations, and an automated or reproducible verification path.

See [Supported Models](docs/supported-models.md) for the full support matrix.

## Installation

Install from PyPI:

```bash
pip install arclm
```

For CPU-only environments, install with the official PyTorch CPU index:

```bash
pip install "arclm[all-cpu]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

For CUDA 12.1:

```bash
pip install "arclm[all-cuda121]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

For local development:

```bash
pip install -e ".[dev,preprocess,hf,web]"
```

ArcLM declares Python `>=3.9,<3.13`.

## Minimal Quick Start

This example uses only public ArcLM APIs and trains a tiny native causal model on CPU.

```python
from pathlib import Path
import tempfile

from arclm import DataProcessor, Tokenizer, load_model, train_model

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    raw_path = root / "records.jsonl"
    train_path = root / "train.txt"
    model_path = root / "model.pth"

    raw_path.write_text(
        '{"text": "ArcLM prepares language model data."}\n'
        '{"text": "Clean records make training easier."}\n',
        encoding="utf-8",
    )

    dataset = (
        DataProcessor.load(raw_path)
        .clean()
        .filter(lambda row: len(row.get("text", "")) > 10)
        .transform(format="pretraining")
    )

    tokenizer = Tokenizer(max_vocab=64)
    tokenizer.build(" ".join(row["text"] for row in dataset.samples))
    tokenized = dataset.tokenize(tokenizer)
    assert all("tokens" in row for row in tokenized.samples)

    train_path.write_text(
        " ".join(row["text"] for row in dataset.samples) * 24,
        encoding="utf-8",
    )

    train_model(
        mode="pretrain",
        data=str(train_path),
        output=str(model_path),
        tokenizer_type="word",
        max_vocab=64,
        embed_dim=16,
        num_blocks=1,
        block_size=8,
        batch_size=2,
        num_epochs=1,
        validation_split=0.0,
        training_log_interval=0,
        device="cpu",
    )

    loaded = load_model(model_path, device="cpu")
    print(loaded.predict("ArcLM", max_new_tokens=4, top_k=3))
```

## Documentation

- [Documentation home](docs/index.md)
- [Getting Started](docs/quick-start.md)
- [Data Preparation Guide](docs/data-guide/loading-data.md)
- [Model Loading Guide](docs/model-guide/loading-models.md)
- [API Reference](docs/api-reference/index.md)
- [CLI Reference](docs/cli-reference.md)
- [Migration Guide](docs/migration-guide.md)
- [Production Readiness](docs/production-readiness.md)
- [Roadmap](docs/roadmap.md)

## Examples

Local examples are in [examples](examples/README.md). Start with:

```bash
python examples/01_quickstart.py
python examples/03_data_processing.py
python examples/11_inference.py
```

Examples that use Hugging Face models may download model files and need optional dependencies:

```bash
pip install -e ".[hf,peft]"
python examples/08_huggingface_sft.py
```

## Honest Limitations

- ArcLM native models are compact GPT-style models, not production-scale LLM architectures.
- Hugging Face model loading is limited to causal language models through `AutoModelForCausalLM`.
- Qwen examples are reproducible examples, not automated release certification.
- Encoder-only and seq2seq models are out of scope for the current public workflow.
- The CLI still contains older pathways that need consolidation with `train_model`.
- Production readiness requires stronger validation, typed configs, full CI coverage, and a release/deprecation policy.

## Contributing

See [Contributing](docs/contributing.md). Contributions should keep ArcLM focused on data-first causal-language-model workflows and should include tests or reproducible examples for new public behavior.

## License

ArcLM is released under the [Apache License 2.0](LICENSE).
