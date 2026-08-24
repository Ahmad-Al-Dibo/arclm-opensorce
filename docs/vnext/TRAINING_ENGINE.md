# ArcLM Training Engine

Status: IMPLEMENTED FOR NATIVE NEXT-TOKEN TRAINING.

`arclm.core.TrainingEngine` owns the training lifecycle for the vNext public
Trainer path.

```text
Trainer(model=Model, dataset=Dataset)
  -> Dataset.prepare()
  -> TrainingEngine
  -> TrainingStrategy
  -> TorchBackend
```

## Responsibilities

The engine owns epoch/step lifecycle, forward/loss execution through strategy,
backward execution, optimizer stepping, scheduler stepping, gradient clipping,
metrics, checkpoint hooks, validation hooks and runtime device transfer.

Torch remains the active backend for tensors/autograd/optimizers through
`TorchBackend`.

## Strategies

| Strategy | Mode | Behavior |
| --- | --- | --- |
| `PretrainStrategy` | `pretrain` | Next-token causal LM loss. |
| `FullFineTuneStrategy` | `full_finetune` | Same current loss loop with all parameters trainable. |
| `AdapterStrategy` | `adapter` or `lora` | Freezes base parameters and trains attached LoRA tensors. |

## Compatibility

The high-level vNext path no longer calls `build_trainer()` or
`arclm.trainer.Trainer`. The positional legacy constructor remains available as
a transitional public API.
