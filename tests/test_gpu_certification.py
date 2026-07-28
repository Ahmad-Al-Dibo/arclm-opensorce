import pytest
import torch

from arclm.resources import DeviceConfig


pytestmark = [pytest.mark.gpu, pytest.mark.gpu_certification]


def test_cuda_device_config_requires_real_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is not available in this environment.")
    selection = DeviceConfig(device="cuda:0", precision="auto").resolve()
    assert selection.selected_device == "cuda:0"
    assert selection.selected_precision in {"float16", "float32", "bfloat16"}
