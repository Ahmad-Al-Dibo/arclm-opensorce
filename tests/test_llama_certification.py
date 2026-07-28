import pytest

from arclm.certification import certify_model_family
from arclm.models import inspect_model_support


pytestmark = [
    pytest.mark.integration,
    pytest.mark.hf,
    pytest.mark.transformers,
    pytest.mark.slow,
    pytest.mark.cpu,
    pytest.mark.model_certification,
]


def test_tiny_llama_cpu_certification_minimal_step():
    source = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    support = inspect_model_support(source, device="cpu", trust_remote_code=False)
    assert support.support_level == "experimental"
    assert support.causal_lm_compatible

    report = certify_model_family("llama", source, device="cpu", max_steps=1)
    assert report.cpu_certification == "certified"
    assert report.training_certification == "minimal_step"
    assert report.gpu_certification == "untested"
