# ArcLM Simple Interface

[![PyPI](https://img.shields.io/pypi/v/arclm.svg)](https://pypi.org/project/arclm/)
[![Python](https://img.shields.io/pypi/pyversions/arclm.svg)](https://pypi.org/project/arclm/)
[![License](https://img.shields.io/pypi/l/arclm.svg)](LICENSE)
[![Status](https://img.shields.io/badge/release-0.6.0-blue.svg)](docs/VERSIONING.md)

ArcLM now includes a packaged web inference interface that can be started from the library, used from Python code, and used to save loaded models locally.

## Install Web Support

```bash
pip install "arclm[web]"
```

## Install Everything

CPU install with the official PyTorch CPU index:

```bash
pip install "arclm[all-cpu]" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

CUDA 12.1 install with the official PyTorch CUDA index:

```bash
pip install "arclm[all-cuda121]" --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
```

The `--index-url` part is passed to pip at install time. It cannot be stored inside `pyproject.toml` extras.

## Run From The CLI

Start the simple interface with:

```bash
python -m arclm --run simple-interface
```

Optional server settings:

```bash
python -m arclm --run simple-interface --host 127.0.0.1 --port 5000 --no-debug
```

## Run From Python

```python
from arclm import run_simple_interface

run_simple_interface()
```

You can also create the Flask app directly:

```python
from arclm import create_simple_interface_app

app = create_simple_interface_app()
```

## Save Models Locally

The interface includes:

- `Model source`
- `Load`
- `Save path`
- `Save mode`
- `Overwrite`
- `Save local`

Default save path:

```text
models/saved
```

Override it with:

```bash
MODEL_SAVE_PATH=models/local python -m arclm --run simple-interface
```

or:

```bash
SAVE_PATH=models/local python -m arclm --run simple-interface
```

When `Save local` succeeds, the interface updates `Model source` to the saved local path so the model can be loaded from disk next time.

## Save API

```http
POST /model/save
```

Example payload:

```json
{
  "model_source": "Qwen/Qwen3-0.6B",
  "save_path": "models/saved",
  "save_mode": "auto",
  "overwrite": false
}
```

`save_mode: "auto"` uses ArcLM's existing save logic for native ArcLM checkpoints, Hugging Face full models, and LoRA adapters.
