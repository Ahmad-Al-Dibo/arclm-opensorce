import pytest

from arclm.models import inspect_model_support
from arclm.supported_models import OFFICIAL


pytestmark = pytest.mark.transformers


def test_inspect_model_support_rejects_non_causal_task():
    report = inspect_model_support("gpt2", task="embedding")

    assert report.is_supported is False
    assert report.errors


def test_inspect_tiny_gpt2_support_from_config_and_tokenizer():
    report = inspect_model_support("hf-internal-testing/tiny-random-gpt2", device="cpu", trust_remote_code=False)

    assert report.support_level == OFFICIAL
    assert report.causal_lm_compatible
    assert report.tokenizer_available
    assert report.model_type == "gpt2"
