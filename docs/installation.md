# Installation

ArcLM declares Python `>=3.9,<3.13`.

## Core

```bash
pip install arclm
```

## CPU

```bash
pip install "arclm[all-cpu]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

## CUDA 12.1

```bash
pip install "arclm[all-cuda121]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

## Optional Features

```bash
pip install "arclm[preprocess]"
pip install "arclm[hf]"
pip install "arclm[peft]"
pip install "arclm[web]"
pip install "arclm[docs]"
```

## Development

```bash
pip install -e ".[dev,preprocess,hf,web]"
python -m pytest tests
mkdocs build --strict
```

Current validation in this repository was blocked by an unsupported local Python 3.14/Torch environment. See [Production Readiness](production-readiness.md).

