# ArcLM

ArcLM is a compact Python framework for causal language-model workflows. The
vNext direction is ArcLM-owned: model architecture, tokenizer contracts, model
loading, training, fine-tuning, runtime behavior, and `.arcmodel` artifacts are
defined by ArcLM rather than by Hugging Face Transformers.

PyTorch is currently used as the tensor/autograd/runtime backend. `safetensors`
is used for model and adapter tensor storage.

## Install

```bash
pip install -e .
```

For development and documentation work:

```bash
pip install -e ".[dev]"
```

ArcLM supports Python `>=3.9,<3.13`.

## Student / Lab API

```python
from arclm import Lab

lab = Lab()
dataset = lab.dataset(text="alpha beta gamma delta alpha beta gamma delta")
model = lab.model(data=dataset, size="tiny")
history = lab.train(epochs=1, steps=1, shuffle=False)
```

## Professional API

```python
from arclm import Dataset, Model, Runtime, Tokenizer, Trainer
from arclm.training import FineTuningConfig, TrainingConfig

dataset = Dataset.load([{"text": "alpha beta gamma delta alpha beta gamma delta"}])
tokenizer = Tokenizer(strategy="word", max_vocab=16).build(dataset.text())
runtime = Runtime.auto(prefer="cpu")
model = Model.create(
    architecture="arclm-native",
    tokenizer=tokenizer,
    runtime=runtime,
    embed_dim=8,
    block_size=4,
    num_blocks=1,
)

config = TrainingConfig(epochs=1, steps_per_epoch=1, save_artifact=True, artifact_path="model.arcmodel")
history = Trainer(model=model, dataset=dataset, config=config, fine_tuning=FineTuningConfig(method="pretrain")).train()
```

## Research API

```python
from arclm.research import BaseStrategy, NextTokenLoss, TrainingConfig, Trainer

class MyLoss:
    def __call__(self, *, model, batch, engine):
        return NextTokenLoss()(model=model, batch=batch, engine=engine) * 0.5

strategy = BaseStrategy(name="half_loss", loss_function=MyLoss())
history = Trainer(model=model, dataset=dataset, config=TrainingConfig(epochs=1, steps_per_epoch=1), strategy=strategy).train()
```

## Current Support

- Native compact causal language model architecture: `arclm-native-causal-lm`.
- Unified `Tokenizer` facade with native word and character engines.
- Optional lazy SentencePiece tokenizer engine.
- ArcLM-owned training engine with deterministic per-epoch steps, validation,
  checkpoint/resume, callbacks, metrics, progress display, gradient
  accumulation, and early stopping.
- Full fine-tuning and native LoRA-style adapter fine-tuning.
- `.arcmodel` model artifacts and `.arcadapter` adapter artifacts.
- Three API levels over the same engine: Student, Professional, Research.

## Planned

- Optional Hugging Face compatibility adapters.
- Additional ArcLM-native model families.
- ArcLM-native BPE, WordPiece, and Unigram tokenizer engines.
- Richer artifact migration/version policies.

## Examples

The files in `examples/` are executable and covered by tests:

- `01_student_lab.py`
- `02_professional_training.py`
- `03_lora_finetuning.py`
- `04_save_load_arcmodel.py`
- `05_research_custom_loss.py`

## Development

```bash
python -m pytest tests
python -m mkdocs build --strict
```

## License

ArcLM is released under the [Apache License 2.0](LICENSE).
