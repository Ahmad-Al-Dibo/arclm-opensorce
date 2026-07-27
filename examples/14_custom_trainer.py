"""Custom trainer: override compute_loss for label smoothing."""

from pathlib import Path
import sys

import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import Tokenizer, Trainer, build_model, build_trainer, create_config, create_dataloader


class LabelSmoothingTrainer(Trainer):
    def compute_loss(self, logits, y, loss_mask=None):
        batch_size, steps, vocab_size = logits.shape
        token_loss = F.cross_entropy(
            logits.reshape(batch_size * steps, vocab_size),
            y.reshape(batch_size * steps),
            reduction="none",
            label_smoothing=0.05,
        ).reshape(batch_size, steps)

        if loss_mask is None:
            return token_loss.mean()

        loss_mask = loss_mask.to(token_loss.device, dtype=token_loss.dtype)
        return (token_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def main():
    text = "custom trainers can change the loss while reusing ArcLM batching " * 12
    tokenizer = Tokenizer(max_vocab=100)
    tokenizer.build(text)
    encoded = tokenizer.encode_text(text)

    config = create_config(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=32,
        num_blocks=1,
        block_size=8,
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-3,
        training_log_interval=0,
        device="cpu",
    )
    loader = create_dataloader(encoded, config.block_size, config.batch_size, shuffle=False)
    base = build_trainer(build_model(config), config)
    trainer = LabelSmoothingTrainer(base.model, base.optimizer, base.criterion, config)
    trainer.train(loader, config.num_epochs)

    print("Trained with custom label smoothing loss.")


if __name__ == "__main__":
    main()
