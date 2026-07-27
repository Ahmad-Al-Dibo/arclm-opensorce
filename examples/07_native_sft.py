"""Native SFT: train with response-only loss masks."""

from pathlib import Path
import sys

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arclm import Config, InstructionDataset, Tokenizer, build_model, build_trainer


def main():
    instructions = [
        "Explain ArcLM in one sentence.",
        "What does assistant-only loss mean?",
    ]
    responses = [
        "ArcLM is a compact PyTorch language-model toolkit.",
        "It trains only on assistant response tokens.",
    ]
    training_text = "\n".join(
        f"<|instruction|>\n{instruction}\n<|response|>\n{response}"
        for instruction, response in zip(instructions, responses)
    )

    tokenizer = Tokenizer(
        max_vocab=200,
        user_defined_symbols=["<|instruction|>", "<|response|>"],
    )
    tokenizer.build(training_text)

    config = Config(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=32,
        block_size=64,
        num_blocks=1,
        batch_size=2,
        num_epochs=1,
        learning_rate=1e-3,
        training_log_interval=0,
        device="cpu",
    )

    dataset = InstructionDataset(instructions, responses, tokenizer, config.block_size)
    trainer = build_trainer(build_model(config), config)
    trainer.train(DataLoader(dataset, batch_size=config.batch_size), config.num_epochs)

    print("Trained one native masked-SFT epoch.")


if __name__ == "__main__":
    main()
