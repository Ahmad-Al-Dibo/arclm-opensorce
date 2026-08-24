# ArcLM Examples

Examples are organized by audience level.

Run commands from the repository root.

## Students Level

Start here when learning ArcLM concepts and the public API:

```bash
python examples/students_level/01_quickstart.py
python examples/students_level/02_tokenization.py
python examples/students_level/03_logic_basics.py
python examples/students_level/04_pretraining.py
```

Includes:

- `01_quickstart.py`: train a tiny native `.arcmodel` with `Lab`.
- `02_tokenization.py`: build and use the word tokenizer.
- `03_logic_basics.py`: beginner symbolic logic rules and entailment.
- `04_pretraining.py`: run native pretraining.
- `05_continue_training.py`: continue a compatible native `.arcmodel`.
- `06_finetuning.py`: next-token fine-tune a native `.arcmodel`.
- `07_native_sft.py`: native LoRA adapter fine-tuning with `Model.finetune`.
- `logics_example.py`: symbolic logic basics.

## Company Level

Use these for practical application workflows and operational inspection:

```bash
python examples/company_level/11_inference.py
python examples/company_level/12_smart_loader.py
python examples/company_level/13_diagnostics.py
python examples/company_level/14_logic_policy_check.py
python examples/company_level/15_load_output_arcmodel.py
python examples/company_level/15_load_output_arcmodel.py --prompt "ArcLM"
```

Includes:

- `08_huggingface_sft.py`: Hugging Face SFT smoke test.
- `11_inference.py`: train, load, and generate from a native `.arcmodel`.
- `12_smart_loader.py`: inspect a local model-like folder.
- `13_diagnostics.py`: inspect top-k predictions.
- `14_logic_policy_check.py`: symbolic deployment-policy checks.
- `15_load_output_arcmodel.py`: load and use `output/*.arcmodel`, including trusted local legacy `.pt` checkpoints.

## Advanced Research Level

These examples expose lower-level or heavier workflows and may use optional
dependencies, downloads, adapters, custom trainers, and larger model references:

```bash
python examples/advanced_research_level/09_lora_sft.py
python examples/advanced_research_level/14_custom_trainer.py
python examples/advanced_research_level/16_logic_capability_reasoning.py
```

Includes:

- `09_lora_sft.py`: Hugging Face LoRA SFT smoke test.
- `14_custom_trainer.py`: customize a `TrainingEngine` strategy.
- `16_logic_capability_reasoning.py`: symbolic reasoning over capability evidence.
- `qwen3_0_6b_sft/`: larger Qwen SFT workflow.

Install optional dependencies for Hugging Face/PEFT examples:

```bash
pip install -e ".[hf,peft]"
```
