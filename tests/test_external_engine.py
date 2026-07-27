from pathlib import Path

import torch

from arclm import (
    Config,
    Tokenizer,
    adapt_for_training,
    build_model,
    build_trainer,
    load_external_model,
    train_model,
)


def _small_config(tmp_path, **overrides):
    values = {
        "embed_dim": 8,
        "block_size": 4,
        "num_blocks": 1,
        "dropout": 0.0,
        "batch_size": 2,
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "training_log_interval": 0,
        "device": "cpu",
        "model_path": str(tmp_path / "model.pth"),
        "vocab_size": 8,
    }
    values.update(overrides)
    return Config(**values)


def test_load_and_adapt_raw_state_dict(tmp_path):
    config = _small_config(tmp_path)
    model = build_model(config, vocab_size=config.vocab_size)
    path = tmp_path / "raw_state.pth"
    torch.save(model.state_dict(), path)

    loaded = load_external_model(path)
    bundle = adapt_for_training(loaded, target_config=config, strict=True)

    assert loaded.source_type == "state_dict"
    assert bundle.config.vocab_size == config.vocab_size
    logits = bundle.model(torch.randint(0, config.vocab_size, (1, config.block_size)))
    assert tuple(logits.shape) == (1, config.block_size, config.vocab_size)


def test_load_native_arclm_checkpoint_with_tokenizer_check(tmp_path):
    tokenizer = Tokenizer(max_vocab=8)
    tokenizer.build("arc lm arc data model")
    config = _small_config(tmp_path, vocab_size=tokenizer.get_vocab_size())
    model = build_model(config, vocab_size=config.vocab_size)
    trainer = build_trainer(model, config)
    trainer.save(
        config,
        vocab=tokenizer.vocab,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        tokenizer_metadata=tokenizer.to_checkpoint(),
    )

    loaded = load_external_model(config.model_path)
    bundle = adapt_for_training(
        loaded,
        target_config=config,
        tokenizer=tokenizer,
        require_tokenizer_match=True,
        strict=True,
    )

    assert loaded.source_type == "arclm"
    assert loaded.stoi == tokenizer.stoi
    assert bundle.missing_keys == []


def test_train_model_pretrain_writes_checkpoint_and_metrics(tmp_path):
    data_path = tmp_path / "data.txt"
    data_path.write_text(
        "arc lm trains on compact text data. " * 12,
        encoding="utf-8",
    )
    output_path = tmp_path / "trained.pth"
    metrics_path = tmp_path / "metrics.jsonl"

    result = train_model(
        mode="pretrain",
        data=str(data_path),
        output=str(output_path),
        embed_dim=8,
        block_size=4,
        num_blocks=1,
        batch_size=2,
        num_epochs=1,
        max_vocab=32,
        training_log_interval=0,
        validation_split=0.0,
        metrics_log_path=str(metrics_path),
        device="cpu",
    )

    assert result.mode == "pretrain"
    assert Path(result.model_path).exists()
    assert metrics_path.exists()
    assert "epoch_completed" in metrics_path.read_text(encoding="utf-8")
