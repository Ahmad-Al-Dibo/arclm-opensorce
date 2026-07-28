# Known Limitations

- Native `ArcLM` is compact and educational; it is not a full-scale LLM stack.
- The model uses single-head causal self-attention.
- Legacy configuration templates may still contain `num_heads`, but the native model does not consume it and the current `arclm train` CLI no longer exposes the unused option.
- Hugging Face model support is scoped to `AutoModelForCausalLM`.
- LoRA requires PEFT and a compatible base model.
- Quantization flags are passed to Transformers but not certified by ArcLM tests.
