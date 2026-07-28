# Training API

| API | Import path | Purpose | Parameters | Returns | Stability |
| --- | --- | --- | --- | --- | --- |
| `train_model` | `arclm.train_model` | Native pretrain/finetune/continue workflow. | `mode`, `data`, `output`, `checkpoint=None`, `config=None`, `tokenizer=None`, `**config_overrides` | `TrainingResult` | Stable-ish |
| `TrainingResult` | `arclm.TrainingResult` | Native training result. | dataclass fields | Result | Stable-ish |
| `Trainer` | `arclm.Trainer` | Native PyTorch training loop. | `model`, `optimizer`, `criterion`, `config` | Trainer | Stable-ish |
| `build_trainer` | `arclm.build_trainer` | Build optimizer/loss/trainer. | `model`, `config`, `event_logger=None` | `Trainer` | Stable-ish |
| `train_sft` | `arclm.train_sft` | Hugging Face SFT wrapper. | `model`, `dataset`, `output_dir`, SFT options | `SFTTrainingResult` | Experimental |
| `SFTTrainingResult` | `arclm.SFTTrainingResult` | SFT result metadata. | dataclass fields | Result | Experimental |
| `InstructionDataset` | `arclm.InstructionDataset` | Native masked instruction dataset. | `instructions`, `responses`, `tokenizer`, `block_size` | Dataset | Stable-ish |
| `create_instruction_dataloader` | `arclm.create_instruction_dataloader` | Build instruction DataLoader. | instructions/responses/tokenizer/settings | `DataLoader` | Stable-ish |
| `UnifiedPipeline` | `arclm.UnifiedPipeline` | Older training pipeline abstraction. | `config`, `mode` | Pipeline | Experimental/legacy |

`train_sft` currently implements only `backend="huggingface"`.

