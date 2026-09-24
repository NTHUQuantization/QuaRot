import torch

from e2e.qwen3_32b_td_numeric import SafeSelectedHiddenCollector
from e2e.speculative import _inverse_hadamard_cuda


def test_power_of_two_prescale_matches_unscaled_transform_without_overflow():
    if not torch.cuda.is_available():
        return
    generator = torch.Generator(device="cuda").manual_seed(31)
    value = torch.randn(
        2, 13, 5120, generator=generator, device="cuda", dtype=torch.float16
    ) * 400
    restored = (
        _inverse_hadamard_cuda(value / SafeSelectedHiddenCollector.prescale).float()
        * SafeSelectedHiddenCollector.prescale
    )
    reference = _inverse_hadamard_cuda(value.float())
    assert torch.isfinite(restored).all()
    relative_rmse = (
        (restored - reference).square().mean().sqrt()
        / reference.square().mean().sqrt()
    )
    assert float(relative_rmse) < 0.002
