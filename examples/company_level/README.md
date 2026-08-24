# Company Level Examples

Practical examples for application teams: inference, model-source inspection,
diagnostics, and Hugging Face SFT smoke workflows.

Run from the repository root:

```bash
python examples/company_level/11_inference.py
python examples/company_level/14_logic_policy_check.py
python examples/company_level/15_load_output_arcmodel.py
python examples/company_level/15_load_output_arcmodel.py --prompt "ArcLM"
python examples/company_level/15_load_output_arcmodel.py output/MiniGPT.arcmodel --prompt "Hello"
python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel
python examples/company_level/17_pretrain_3m_gtx1050ti.py
```

The fine-tuning example downloads Tiny Shakespeare for pre-fine-tuning and
Stanford Alpaca-style instruction records for SFT, then writes prepared files
under `data/`.

Full usage:

```bash
# Full default run: download data, prepare it, pre-fine-tune, SFT fine-tune, save outputs.
python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel

# Quick smoke run.
python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel --pretrain-max-chars 512 --sft-max-records 1 --batch-size 64 --block-size 16 --max-new-tokens 3

# Only pre-fine-tuning on plain text.
python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel --skip-sft

# Only SFT fine-tuning on instruction JSONL.
python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel --skip-pretrain

# Custom output paths.
python examples/company_level/16_finetune_output_arcmodel.py output/MiniGPT.arcmodel --save-pretrained-to output/MiniGPT-pre.arcmodel --save-sft-to output/MiniGPT-sft.arcmodel
```

GTX 1050 Ti 4GB pretraining:

```bash
# Default >3M parameter run.
python examples/company_level/17_pretrain_3m_gtx1050ti.py

# Use your own corpus.
python examples/company_level/17_pretrain_3m_gtx1050ti.py --data data/my_corpus.txt --output output/my-3m.arcmodel

# If CUDA runs out of memory.
python examples/company_level/17_pretrain_3m_gtx1050ti.py --batch-size 2 --block-size 32
```
