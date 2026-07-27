# Fine-tuning Qwen3-0.6B With ArcLM

This example shows the full ArcLM SFT workflow for `Qwen/Qwen3-0.6B`:

1. Load the base Hugging Face model.
2. Generate baseline answers.
3. Train on a tiny chat-style SFT dataset with `from arclm import train_sft`.
4. Save a LoRA adapter by default.
5. Load the adapter on top of the base model.
6. Generate fine-tuned answers.
7. Run a small functional benchmark.

This is a demonstration, not a serious model evaluation. The dataset has only four rows so the model may memorize wording rather than learn a robust behavior.

## Requirements

Install ArcLM and the Hugging Face/PEFT stack:

```bash
pip install -e .[peft]
```

For Qwen3, use `transformers>=4.51`. The example does not require `bitsandbytes`; quantized loading is not used.

## Windows PowerShell

```powershell
.\.venv\Scripts\python.exe examples/qwen3_0_6b_sft/test_base_model.py
.\.venv\Scripts\python.exe examples/qwen3_0_6b_sft/train_qwen3_0_6b_sft.py
.\.venv\Scripts\python.exe examples/qwen3_0_6b_sft/test_finetuned_model.py
.\.venv\Scripts\python.exe examples/qwen3_0_6b_sft/benchmark_base_vs_finetuned.py
```

## Cross-platform

```bash
python examples/qwen3_0_6b_sft/test_base_model.py
python examples/qwen3_0_6b_sft/train_qwen3_0_6b_sft.py
python examples/qwen3_0_6b_sft/test_finetuned_model.py
python examples/qwen3_0_6b_sft/benchmark_base_vs_finetuned.py
```

## Base Model Test

```bash
python examples/qwen3_0_6b_sft/test_base_model.py
```

This loads `Qwen/Qwen3-0.6B`, applies the tokenizer chat template, tries `enable_thinking=False`, prints only generated text, and writes:

```text
examples/qwen3_0_6b_sft/output/base_outputs.jsonl
```

## Fine-tuning

```bash
python examples/qwen3_0_6b_sft/train_qwen3_0_6b_sft.py
```

The script calls ArcLM's public API:

```python
from arclm import train_sft

result = train_sft(
    model="Qwen/Qwen3-0.6B",
    dataset="examples/qwen3_0_6b_sft/data/sample_sft.jsonl",
    output_dir="examples/qwen3_0_6b_sft/output/qwen3_0_6b_sft_lora",
    backend="huggingface",
    assistant_only_loss=True,
    use_lora=True,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_epochs=1,
    max_length=1024,
    dtype="auto",
    device_map="auto",
)
```

By default the script saves a PEFT LoRA adapter under:

```text
examples/qwen3_0_6b_sft/output/qwen3_0_6b_sft_lora
```

## Fine-tuned Model Test

```bash
python examples/qwen3_0_6b_sft/test_finetuned_model.py
```

This loads the base model, attaches the LoRA adapter if `adapter_config.json` exists, generates answers for the same prompts, and writes:

```text
examples/qwen3_0_6b_sft/output/finetuned_outputs.jsonl
```

## Benchmark

```bash
python examples/qwen3_0_6b_sft/benchmark_base_vs_finetuned.py
```

The benchmark compares keyword matches and answer length. It prints:

```text
This is a small functional benchmark, not a full model evaluation.
```

It also writes:

```text
examples/qwen3_0_6b_sft/output/benchmark_results.json
```

## Generated Files

- `output/base_outputs.jsonl`
- `output/qwen3_0_6b_sft_lora/`
- `output/finetuned_outputs.jsonl`
- `output/benchmark_results.json`

## LoRA

LoRA trains small low-rank adapter matrices while leaving most base model weights frozen. It is much lighter than full fine-tuning and is the default for this example.

To disable LoRA and run full fine-tuning:

```bash
python examples/qwen3_0_6b_sft/train_qwen3_0_6b_sft.py --no-lora
```

Only do this if you have enough GPU memory. Full fine-tuning saves a full Hugging Face model instead of an adapter.

## Assistant-only Loss

With `assistant_only_loss=True`, ArcLM tokenizes the full conversation but labels only assistant answer tokens. User and system tokens remain context, but their labels are set to `-100` so they do not contribute to cross-entropy loss.

## Replacing The Dataset

Replace `data/sample_sft.jsonl` with JSONL rows using either:

```json
{"messages":[{"role":"user","content":"Question"},{"role":"assistant","content":"Answer"}]}
```

or:

```json
{"instruction":"Question","output":"Answer"}
```

Keep examples clean, factual, and consistent with the behavior you want the model to learn.

## Troubleshooting

`CUDA out of memory`

Use LoRA, keep `batch_size=1`, reduce `max_length`, reduce `max_new_tokens` in test scripts, or use a smaller model.

`Missing PEFT`

Install `peft` with `pip install -e .[peft]` or run full fine-tuning with `--no-lora`.

`Missing transformers`

Install ArcLM runtime dependencies with `pip install -e .`. Qwen3 needs `transformers>=4.51`.

`Hugging Face download warnings`

The first run downloads model files to the Hugging Face cache. Check your network connection, disk space, and any required Hugging Face authentication.

`Qwen chat template issues`

Upgrade `transformers`. The scripts try `enable_thinking=False` and retry without it if the tokenizer does not accept that argument.
