from pathlib import Path

import torch

from arclm import (
    Config,
    InstructionDataset,
    Tokenizer,
    build_model,
    build_trainer,
    create_instruction_dataloader,
    tokenizer_from_checkpoint,
)


def _tokenizer():
    tokenizer = Tokenizer(
        max_vocab=32,
        user_defined_symbols=["<|instruction|>", "<|response|>"],
    )
    tokenizer.build("<|instruction|> say hi <|response|> hello world")
    return tokenizer


def test_instruction_dataset_masks_shifted_response_labels():
    tokenizer = _tokenizer()
    dataset = InstructionDataset(
        instructions=["say hi"],
        responses=["hello world"],
        tokenizer=tokenizer,
        block_size=8,
    )

    item = dataset[0]
    active_labels = [
        int(label)
        for label, active in zip(item["y"].tolist(), item["mask"].tolist())
        if active == 1.0
    ]

    assert tokenizer.decode(active_labels) == ["hello", "world"]


def test_trainer_accepts_masked_instruction_batches():
    tokenizer = _tokenizer()
    loader = create_instruction_dataloader(
        instructions=["say hi"],
        responses=["hello world"],
        tokenizer=tokenizer,
        block_size=8,
        batch_size=1,
        shuffle=False,
    )
    config = Config(
        embed_dim=8,
        block_size=8,
        num_blocks=1,
        batch_size=1,
        num_epochs=1,
        vocab_size=tokenizer.get_vocab_size(),
        learning_rate=1e-3,
        training_log_interval=0,
        device="cpu",
    )
    model = build_model(config, config.vocab_size)
    trainer = build_trainer(model, config)

    trainer.train(loader, config.num_epochs)

    assert len(trainer.train_losses) == 1
    assert torch.isfinite(torch.tensor(trainer.train_losses[0]))


def test_tokenizer_from_checkpoint_restores_word_tokenizer(tmp_path):
    tokenizer = _tokenizer()
    config = Config(
        model_path=str(tmp_path / "model.pth"),
        embed_dim=8,
        block_size=8,
        num_blocks=1,
        vocab_size=tokenizer.get_vocab_size(),
        device="cpu",
    )
    model = build_model(config, config.vocab_size)
    trainer = build_trainer(model, config)
    trainer.save(
        config,
        vocab=tokenizer.vocab,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        tokenizer_metadata=tokenizer.to_checkpoint(),
    )

    restored = tokenizer_from_checkpoint(Path(config.model_path))

    assert restored.stoi == tokenizer.stoi
    assert restored.encode_text("hello world") == tokenizer.encode_text("hello world")
