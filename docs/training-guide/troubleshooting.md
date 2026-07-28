# Training Troubleshooting

- If the dataset is too short for `block_size`, reduce `block_size` or add data.
- If tokenizer compatibility fails, use the tokenizer stored in the checkpoint or rebuild the run from scratch.
- If CUDA runs out of memory, reduce `batch_size`, `block_size`, model size, or use CPU for small examples.
- If SFT cannot find assistant labels, set `assistant_only_loss=False` or inspect the tokenizer chat template.

