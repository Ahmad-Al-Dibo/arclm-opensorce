"""Custom training strategy: label smoothing with the ArcLM TrainingEngine."""

from pathlib import Path
import sys

import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arclm import Dataset, Model, Runtime
from arclm.core import PretrainStrategy, TrainingEngine, TrainingEngineConfig


class LabelSmoothingStrategy(PretrainStrategy):
    """Next-token loss with label smoothing."""

    name = "label_smoothing_pretrain"

    def loss(self, *, model, batch, engine):
        inputs, targets = engine.unpack_next_token_batch(batch)
        logits = model(inputs)
        batch_size, steps, vocab_size = logits.shape
        return F.cross_entropy(
            logits.reshape(batch_size * steps, vocab_size),
            targets.reshape(batch_size * steps),
            label_smoothing=0.05,
        )


def main():
    text = "custom strategies can change the loss while reusing ArcLM batching " * 16
    dataset = Dataset.load([{"text": text}])
    prepared = dataset.prepare(block_size=8, batch_size=2, shuffle=False)
    runtime = Runtime.auto(prefer="cpu")
    model = Model.create(
        architecture="arclm-native",
        tokenizer=prepared.tokenizer,
        runtime=runtime,
        embed_dim=16,
        block_size=8,
        num_blocks=1,
        dropout=0.0,
    )

    engine = TrainingEngine(
        runtime=runtime,
        config=TrainingEngineConfig(epochs=1, learning_rate=1e-3),
    )
    result = engine.fit(
        model=model.model,
        dataloader=prepared.train_loader,
        strategy=LabelSmoothingStrategy(),
    )

    print(f"Strategy: {result.strategy}")
    print(f"Losses: {result.train_losses}")


if __name__ == "__main__":
    main()
