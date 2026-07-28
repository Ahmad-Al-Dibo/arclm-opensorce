from pathlib import Path

import torch

from arclm import Config, Tokenizer, build_model, build_trainer, inspect_checkpoint


def test_checkpoint_inspection_distinguishes_untrusted_from_trusted(tmp_path):
    tokenizer = Tokenizer(max_vocab=8)
    tokenizer.build("arc lm data")
    config = Config(
        model_path=str(tmp_path / "model.pth"),
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=8,
        block_size=4,
        num_blocks=1,
        device="cpu",
    )
    trainer = build_trainer(build_model(config, config.vocab_size), config)
    trainer.save(
        config,
        vocab=tokenizer.vocab,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        tokenizer_metadata=tokenizer.to_checkpoint(),
    )

    untrusted = inspect_checkpoint(config.model_path, trusted=False)
    trusted = inspect_checkpoint(config.model_path, trusted=True)

    assert untrusted.exists
    assert untrusted.load_attempted is False
    assert untrusted.warnings
    assert trusted.is_loadable
    assert trusted.is_native_arclm
    assert trusted.has_tokenizer_metadata


def test_checkpoint_inspection_reports_missing_and_corrupt_files(tmp_path):
    missing = inspect_checkpoint(tmp_path / "missing.pth", trusted=True)
    assert missing.errors

    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_text("not a checkpoint", encoding="utf-8")
    report = inspect_checkpoint(corrupt, trusted=True)
    assert report.errors


def test_checkpoint_inspection_warns_on_version_mismatch(tmp_path):
    path = tmp_path / "old.pth"
    torch.save(
        {
            "model_state_dict": {},
            "config": {"arclm_version": "0.0.1"},
            "vocab_size": 1,
        },
        path,
    )

    report = inspect_checkpoint(path, trusted=True)

    assert any("0.0.1" in warning for warning in report.warnings)
