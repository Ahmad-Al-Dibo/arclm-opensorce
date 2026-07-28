# Metrics

Training history is available from `TrainingResult.history`.

```python
result = train_model("pretrain", "data.txt", "model.pth", num_epochs=1)
print(result.history)
```

Diagnostic helpers include `calculate_metrics`, `calculate_perplexity`, `export_metrics_to_json`, and `export_metrics_to_markdown`.

