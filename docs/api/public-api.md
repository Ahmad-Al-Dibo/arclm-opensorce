# Public API

ArcLM's top-level API is intentionally small:

```python
from arclm import (
    Architecture,
    ArchitectureCapabilities,
    ArchitectureKind,
    ArchitectureRegistry,
    Dataset,
    Lab,
    Model,
    Runtime,
    Tokenizer,
    Trainer,
    architectures,
)
```

## Student

- `Lab`

## Professional

- `Dataset`
- `Tokenizer`
- `Model`
- `Runtime`
- `Trainer`
- `TrainingConfig`
- `FineTuningConfig`

## Research

Import intentional extension points from `arclm.research`.

```python
from arclm.research import BaseStrategy, NextTokenLoss, TrainingEngine, TokenizerEngine
```

The research namespace is the supported place to plug in custom strategies,
losses, tokenizers, evaluators, callbacks, checkpoint managers, architecture
definitions, and backend/runtime components.
