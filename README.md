# ArcLM

ArcLM is an ArcLM-first, self-driving framework for causal language-model work.

It is no longer positioned as a PyTorch toolkit. ArcLM is its own framework, with PyTorch, Hugging Face, and related tools treated as backends or dependencies where they are useful. The public route is ArcLM's route.

ArcLM is moving from a collection of helpful LLM utilities into a framework that can carry the whole route: inspect the data, validate the configuration, prepare the dataset, choose the model path, run training or fine-tuning, evaluate the result, save artifacts, and report what happened.

The goal is simple: give developers a strong framework layer with the discipline of the most powerful production libraries, while still allowing proven engines underneath.

## The Route

```text
Raw data
-> inspect
-> clean
-> validate
-> tokenize
-> load model
-> train or fine-tune
-> evaluate
-> generate
-> save and report
```

ArcLM calls this direction self-driving because the framework is designed to make every stage explicit, checked, reproducible, and connected under ArcLM's own APIs. You still own the decisions. ArcLM owns the boring failure points: bad records, hidden tokenizer mismatches, unsafe checkpoints, unclear device choices, missing reports, and fragile training scripts.

## What It Does

- Data loading, cleaning, validation, splitting, sharding, and quality reports.
- Tokenizer building, dataset tokenization, and deterministic tokenization caches.
- Typed workflow configuration with migration and redacted effective config export.
- Safe checkpoint inspection, hash verification, and trusted loading policies.
- Native compact causal-model training with ArcLM-owned `.arcmodel` artifacts.
- Hugging Face causal-LM inspection, loading, inference, and SFT paths through ArcLM APIs.
- Evaluation, generation, run metadata, diagnostics, reproducibility fingerprints, and CLI tools.

## Install

```bash
pip install arclm
```

CPU-only PyTorch:

```bash
pip install "arclm[all-cpu]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

CUDA 12.1:

```bash
pip install "arclm[all-cuda121]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

Development:

```bash
pip install -e ".[dev,preprocess,hf,web]"
```

ArcLM supports Python `>=3.9,<3.13`.

## Quick Start

Create a small workflow config:

```json
{
  "run": {"name": "demo", "output_dir": "runs"},
  "data": {"path": "data/train.jsonl", "format": "jsonl", "schema": "text"},
  "model": {"source": "hf-internal-testing/tiny-random-gpt2", "device": "cpu"},
  "training": {"enabled": false}
}
```

Validate the route before running real work:

```bash
arclm run arclm.json --dry-run
arclm doctor
```

Or use the native Python API:

```python
from arclm import DataProcessor, Lab, Runtime

dataset = (
    DataProcessor.load("data/train.jsonl")
    .clean()
    .transform(format="pretraining")
)

lab = Lab(runtime=Runtime.auto(prefer="cpu"))
model = lab.pretrain("data/train.txt", size="tiny", epochs=1)
model.save("model.arcmodel", overwrite=True)
```

## Status

Current version: `1.0.0`.

ArcLM focuses on causal language models, decoder-only transformer workflows, native compact ArcLM models, and verified Hugging Face `AutoModelForCausalLM` paths. Encoder-only, seq2seq, RLHF, DPO, PPO, reward modeling, vision, audio, and multimodal workflows are outside the current public route.

## Project Files

- [docs/](docs/) - source documentation.
- [site/](site/) - generated documentation site.
- [examples/](examples/) - runnable examples.

## Contributing

Contributions should strengthen the self-driving route: clearer validation, safer loading, better reports, stronger model support, cleaner training paths, and tests or reproducible examples for every new public behavior.

## License

ArcLM is released under the [Apache License 2.0](LICENSE).
