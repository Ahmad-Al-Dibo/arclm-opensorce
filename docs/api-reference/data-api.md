# Data API

| API | Import path | Purpose | Parameters | Returns | Stability |
| --- | --- | --- | --- | --- | --- |
| `DataProcessor` | `arclm.DataProcessor` | Load JSON, JSONL, CSV, TXT, or custom datasets. | Static `load(path, format=None, loader=None)` | `ProcessedDataset` | Stable-ish |
| `ProcessedDataset` | `arclm.ProcessedDataset` | In-memory record collection. | `samples`, `source` | Dataset wrapper | Stable-ish |
| `ProcessedDataset.clean` | `arclm.ProcessedDataset.clean` | Normalize whitespace in text fields. | `text_keys=None` | `ProcessedDataset` | Stable-ish |
| `ProcessedDataset.filter` | `arclm.ProcessedDataset.filter` | Keep rows accepted by a predicate. | `predicate` | `ProcessedDataset` | Stable-ish |
| `ProcessedDataset.transform` | `arclm.ProcessedDataset.transform` | Write formatted `text` field. | `format`, `mapping`, `template`, `text_fields`, `tokenizer` | `ProcessedDataset` | Stable-ish |
| `ProcessedDataset.tokenize` | `arclm.ProcessedDataset.tokenize` | Add `tokens` field. | `tokenizer`, `text_key="text"` | `ProcessedDataset` | Stable-ish |
| `ProcessedDataset.split` | `arclm.ProcessedDataset.split` | Shuffle and split records. | `train`, `validation`, `test`, `seed` | `dict[str, list]` | Stable-ish |
| `DataBundle` | `arclm.DataBundle` | Prepared native training tokens/loaders. | dataclass fields | Bundle | Stable-ish |
| `read_tokens` | `arclm.read_tokens` | Read lowercase tokens from text file. | `path`, `limit` | `list` | Stable-ish |
| `load_tokens` | `arclm.load_tokens` | Load configured raw tokens. | `config` | `list` | Stable-ish |
| `split_train_val` | `arclm.split_train_val` | Split token sequence. | `tokens`, `validation_split`, `block_size`, `seed` | `(train, validation)` | Stable-ish |
| `prepare_data` | `arclm.prepare_data` | Build tokenizer, encode tokens, create loaders. | `config`, `existing_tokenizer=None` | `DataBundle` | Stable-ish |
| `TextDataset` | `arclm.TextDataset` | Next-token PyTorch dataset. | `encoded_data`, `block_size` | Dataset | Stable-ish |
| `create_dataloader` | `arclm.create_dataloader` | Build `DataLoader`. | `encoded_data`, `block_size`, `batch_size`, `shuffle` | `DataLoader` | Stable-ish |
| `PreprocessConfig` | `arclm.preprocess.PreprocessConfig` | JSONL preprocessing settings. | dataclass fields | Config | Experimental |
| `PreprocessPipeline` | `arclm.preprocess.PreprocessPipeline` | Clean/filter/report JSONL rows. | `config` | Pipeline | Experimental |
| `TextRecord` | `arclm.TextRecord` | Validated text record. | `text`, `metadata` | Record | Stable-ish |
| `PromptCompletionRecord` | `arclm.PromptCompletionRecord` | Validated prompt/completion record. | `prompt`, `completion`, `metadata` | Record | Stable-ish |
| `InstructionRecord` | `arclm.InstructionRecord` | Validated instruction/input/output record. | `instruction`, `input`, `output`, `metadata` | Record | Stable-ish |
| `ConversationRecord` | `arclm.ConversationRecord` | Validated chat messages record. | `messages`, `metadata` | Record | Stable-ish |
| `validate_records` | `arclm.data.validate_records` / `arclm.validate_records` | Validate batches without filtering. | `records`, `schema`, `strict`, `allow_empty`, duplicate options | `DatasetValidationReport` | Stable-ish |
| `DataPipeline` | `arclm.data.DataPipeline` / `arclm.DataPipeline` | Compose deterministic data-prep operations. | `seed` | Pipeline | Stable-ish |
| `DataPipelineReport` | `arclm.DataPipelineReport` | Structured pipeline execution report. | dataclass fields | Report | Stable-ish |

Example:

```python
from arclm import DataPipeline, DataProcessor

dataset = DataProcessor.load("data.jsonl").clean().transform(format="pretraining")
processed, report = DataPipeline().normalize_text().validate("text").run(dataset.samples)
```
