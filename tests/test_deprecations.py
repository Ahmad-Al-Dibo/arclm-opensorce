import pytest

from arclm import ArcLM, MiniGPT
from arclm.external_inference import load_any_model
from arclm.pipeline import checkpoint_is_compatible_for_tuining, checkpoint_is_compatible_for_tuning


def test_minigpt_warns_when_instantiated():
    with pytest.warns(DeprecationWarning, match="MiniGPT"):
        model = MiniGPT(vocab_size=4, embed_dim=4, block_size=2, num_blocks=1)

    assert isinstance(model, ArcLM)


def test_tuining_typo_warns_and_delegates():
    checkpoint = {
        "config": {"embed_dim": 4, "block_size": 2, "num_blocks": 1},
        "vocab_size": 4,
    }

    class ConfigLike:
        embed_dim = 4
        block_size = 2
        num_blocks = 1
        tokenizer_type = "word"
        sentencepiece_model_type = "bpe"
        weight_decay = 0.0
        dropout = 0.0
        validation_split = 0.0

    expected = checkpoint_is_compatible_for_tuning(checkpoint, ConfigLike(), 4)
    with pytest.warns(DeprecationWarning, match="checkpoint_is_compatible_for_tuining"):
        actual = checkpoint_is_compatible_for_tuining(checkpoint, ConfigLike(), 4)

    assert actual == expected


def test_load_any_model_warns_when_called(tmp_path):
    unsupported = tmp_path / "unsupported.txt"
    unsupported.write_text("not a model", encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="load_any_model"):
        with pytest.raises(Exception):
            load_any_model(unsupported)
