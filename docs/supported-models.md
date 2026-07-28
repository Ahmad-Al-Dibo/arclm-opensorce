# Supported Models

ArcLM support levels are deliberately narrower than "Transformers can load it."

## Support Levels

| Level | Meaning |
| --- | --- |
| Officially supported | ArcLM verifies loading, tokenizer loading, causal-LM behavior, claimed training/inference workflows, limitations, and a reproducible or automated check. |
| Experimentally supported | A real integration path and example exist, but coverage is not strong enough for official support. |
| Compatible but not tested | The architecture may work through a generic backend, but ArcLM has no family-specific verification. |
| Not supported | Outside the current causal-LM workflow or known not to work with public APIs. |

## Model Matrix

| Model family | Example models | Architecture | Status | Training | Inference | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| ArcLM native checkpoint | `ArcLM`, `MiniGPT` | Compact GPT-style decoder-only causal LM | Officially supported | Yes | Yes | Verified by local tests and examples. |
| GPT-2 through Hugging Face | `gpt2`, `hf-internal-testing/tiny-random-gpt2` | Decoder-only causal LM | Officially supported | SFT yes for tiny certification path | Yes | Certified with tiny GPT-2 config, tokenizer, CPU inference, SFT step, save, reload, and post-reload inference. |
| Qwen through Hugging Face | `Qwen/Qwen3-0.6B` | Decoder-only causal LM | Experimental | SFT example exists | Example scripts exist | Not automated in the core test suite. |
| Llama through Hugging Face | `hf-internal-testing/tiny-random-LlamaForCausalLM` | Decoder-only causal LM | Experimental | Minimal CPU SFT step certified for tiny artifact | CPU inference certified for tiny artifact | Large Llama checkpoints, chat-template correctness, and GPU support remain unverified. |
| Mistral, Gemma, Falcon through Hugging Face | `owner/model-id` | Expected decoder-only causal LM | Compatible but not tested | Not verified | Generic path only | Loader may accept them; users must validate behavior. |
| Encoder-only and seq2seq models | `bert-base-uncased`, `t5-small` | Masked LM or encoder-decoder | Not supported | No | No | Out of current project scope. |

## Requirements For Official Support

- Successful model loading.
- Successful tokenizer loading.
- Correct causal-language-model behavior.
- Working inference.
- Working training or fine-tuning where claimed.
- Tested configuration.
- Documented limitations.
- Reproducible example.
- Automated or documented verification.

The same data is available from Python:

```python
from arclm import OFFICIAL, get_supported_models

for record in get_supported_models(OFFICIAL):
    print(record.family, record.architecture)
```
