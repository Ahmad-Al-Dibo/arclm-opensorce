from arclm import (
    COMPATIBLE_UNTESTED,
    EXPERIMENTAL,
    NOT_SUPPORTED,
    OFFICIAL,
    get_model_capability,
    get_supported_models,
    is_model_officially_supported,
)


def test_supported_models_have_required_levels():
    statuses = {record.status for record in get_supported_models()}

    assert OFFICIAL in statuses
    assert EXPERIMENTAL in statuses
    assert COMPATIBLE_UNTESTED in statuses
    assert NOT_SUPPORTED in statuses


def test_native_arclm_is_only_official_family():
    official = get_supported_models(OFFICIAL)

    assert [record.family for record in official] == [
        "ArcLM native checkpoint",
        "GPT-2 through Hugging Face",
    ]
    assert is_model_officially_supported("ArcLM native")
    assert is_model_officially_supported("GPT-2")


def test_supported_model_records_include_verification_and_limits():
    for record in get_supported_models():
        assert record.family
        assert record.architecture
        assert record.example_models
        assert record.known_limitations
        assert record.verification
        assert record.to_dict()["family"] == record.family

    qwen = get_model_capability("Qwen")
    assert qwen.status == EXPERIMENTAL
