# Why ArcLM?

Many language-model experiments fail because the data workflow is informal: records are loaded differently in each script, invalid samples are kept, formatting choices are not recorded, and tokenizers drift from checkpoints.

ArcLM focuses on the workflow around the model:

- Load source data with explicit formats.
- Clean and transform records before training.
- Validate assumptions early.
- Save tokenizer and checkpoint metadata together.
- Keep model support honest and scoped to causal language modeling.

The framework is for developers, researchers, and students who want small, understandable building blocks rather than a large platform that silently chooses behavior.

