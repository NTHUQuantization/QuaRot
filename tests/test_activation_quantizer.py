import pytest
import torch


_HIP = pytest.importorskip("quarot._HIP")
if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires a ROCm/HIP GPU", allow_module_level=True)


def test_activation_quantizer_default_clip_and_scale_contract():
    from quarot.nn import Quantizer

    quantizer = Quantizer()
    inputs = torch.zeros(2, 3, 128, device="cuda", dtype=torch.float16)
    inputs[0, 0] = torch.linspace(-2, 2, 128, device="cuda", dtype=torch.float16)
    packed = quantizer(inputs)

    assert quantizer.input_clip_ratio == 0.9
    assert packed.scales_x.shape == (2, 3, 1)
    assert torch.isfinite(packed.scales_x).all()
    assert (packed.scales_x > 0).all()
