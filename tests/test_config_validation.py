import pytest

from arclm import Config, ConfigurationError, create_config, normalize_device, normalize_precision


def test_config_validate_normalizes_values(tmp_path):
    config = Config(
        data_path=str(tmp_path / "data.txt"),
        model_path=str(tmp_path / "model.pth"),
        tokenizer_path=str(tmp_path / "tok.json"),
        device="auto",
        seed=123,
    )

    assert config.validate() is config
    assert config.device in {"cpu", "cuda"}


def test_config_validation_rejects_bad_values():
    with pytest.raises(ConfigurationError):
        normalize_device("tpu")
    with pytest.raises(ConfigurationError):
        normalize_precision("int2")
    with pytest.raises(ConfigurationError):
        Config(block_size=0).validate()


def test_create_config_still_rejects_unknown_fields():
    with pytest.raises(ValueError):
        create_config(num_heads=4)
