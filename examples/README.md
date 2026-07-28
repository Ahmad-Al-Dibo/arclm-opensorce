# ArcLM Examples

These examples target ArcLM 0.8.0 development APIs. The numbered scripts are maintained public examples and each focuses on one real API path.

Run from the repository root:

```bash
python examples/01_quickstart.py
```

## Local Examples

These create temporary data and run offline:

- `01_quickstart.py`: train a tiny native checkpoint.
- `02_tokenization.py`: build and use the word tokenizer.
- `03_data_processing.py`: load and format JSONL records.
- `04_pretraining.py`: run native pretraining.
- `05_continue_training.py`: continue a compatible native checkpoint.
- `06_finetuning.py`: next-token fine-tune a checkpoint.
- `07_native_sft.py`: native masked instruction tuning.
- `11_inference.py`: train, load, and generate from a checkpoint.
- `12_smart_loader.py`: inspect a local model-like folder.
- `13_diagnostics.py`: inspect top-k predictions.
- `14_custom_trainer.py`: override `Trainer.compute_loss`.

## Optional Examples

These need optional dependencies or network/model downloads:

- `08_huggingface_sft.py`: Hugging Face SFT smoke test.
- `09_lora_sft.py`: Hugging Face LoRA SFT smoke test.
- `10_preprocess_pipeline.py`: JSONL preprocessing with `arclm[preprocess]`.
- `15_custom_hf_sft_loop.py`: custom Hugging Face SFT loop using ArcLM's public SFT helper classes.

## Advanced Workflow

- `qwen3_0_6b_sft/`: larger Qwen SFT workflow. This is an experimental, reproducible example, not an official support guarantee.
