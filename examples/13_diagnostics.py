"""Diagnostics: inspect top-k next-token predictions."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import ArcLM, Tokenizer, format_top_k_predictions, predict_top_k


def main():
    tokenizer = Tokenizer(max_vocab=32)
    tokenizer.build("arc lm learns tokens arc lm predicts tokens")
    model = ArcLM(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=16,
        block_size=8,
        num_blocks=1,
    )

    predictions = predict_top_k(
        model=model,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        block_size=8,
        device=torch.device("cpu"),
        prompt="arc lm",
        k=3,
        tokenizer=tokenizer,
    )

    print(format_top_k_predictions("arc lm", predictions))


if __name__ == "__main__":
    main()
