# Evaluation API

| API | Import path | Purpose | Parameters | Returns | Stability |
| --- | --- | --- | --- | --- | --- |
| `calculate_perplexity` | `arclm.calculate_perplexity` | Convert loss to perplexity. | `loss` | `float` | Stable-ish |
| `calculate_metrics` | `arclm.calculate_metrics` | Evaluate native model on loader. | `model`, `val_loader`, `config`, `device` | `MetricsReport` | Stable-ish |
| `MetricsReport` | `arclm.MetricsReport` | Evaluation metrics. | dataclass fields | Report | Stable-ish |
| `predict_top_k` | `arclm.predict_top_k` | Inspect next-token top-k. | model/token maps/config prompt | `list[TopKPrediction]` | Experimental |
| `build_training_diagnostics_report` | `arclm.build_training_diagnostics_report` | Build diagnostics report. | `model`, `data`, `config` | dict/report | Experimental |
| `export_metrics_to_json` | `arclm.export_metrics_to_json` | Write JSON metrics. | `metrics_report`, `filepath` | `None` | Stable-ish |
| `export_metrics_to_markdown` | `arclm.export_metrics_to_markdown` | Write Markdown metrics. | `metrics_report`, `filepath` | `None` | Stable-ish |

