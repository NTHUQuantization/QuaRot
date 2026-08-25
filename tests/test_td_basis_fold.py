from types import SimpleNamespace

import pytest
import torch

from e2e.benchmark_pard2 import parser as benchmark_parser
from e2e.pard2 import parser as generation_parser
from e2e.speculative import (
    SelectedHiddenCollector,
    TargetFeatEmbedWarp,
    _normalized_hadamard_cpu,
    fold_td_projection_weight,
)


class _ToyDraft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=3)
        self.embedding = torch.nn.Embedding(5, 3)

    def get_input_embeddings(self):
        return self.embedding


class _ToyTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4, bias=False) for _ in range(32)])
        self.model.norm = torch.nn.Linear(4, 4, bias=False)


def test_folded_td_projection_matches_legacy_row_vector_algebra():
    torch.manual_seed(29)
    hidden, output = 16, 7
    signs = torch.where(torch.arange(hidden) % 3 != 0, 1.0, -1.0)
    gamma = torch.linspace(0.5, 1.5, hidden)
    weight = torch.randn(output, 4 * hidden)
    original = [torch.randn(2, 3, hidden) for _ in range(4)]
    rotated = [_normalized_hadamard_cpu(value * signs) for value in original]
    legacy_features = torch.cat((original[0] * gamma, *original[1:]), dim=-1)
    folded_features = torch.cat(rotated, dim=-1)

    folded_weight = fold_td_projection_weight(weight, signs, gamma, hidden)

    torch.testing.assert_close(
        folded_features @ folded_weight.t(),
        legacy_features @ weight.t(), atol=2e-5, rtol=2e-5)
    assert folded_weight.shape == weight.shape
    assert folded_weight.device.type == "cpu"
    assert folded_weight.dtype == torch.float32


def test_folded_td_collector_uses_final_norm_then_raw_tap_order():
    target = _ToyTarget()
    with torch.no_grad():
        target.model.norm.weight.copy_(2 * torch.eye(4))
    collector = SelectedHiddenCollector(
        target, (-1, -8, -16, -24), folded_basis=True)
    x = torch.ones(1, 2, 4)
    outputs = {}
    for index, layer in enumerate(target.model.layers):
        with torch.no_grad():
            layer.weight.copy_(torch.eye(4) * (index + 1))
        outputs[index] = layer(x)
    target.model.norm(outputs[31])

    features = collector.features()

    assert features.shape == (1, 2, 16)
    torch.testing.assert_close(features[..., :4], outputs[31] * 2)
    torch.testing.assert_close(features[..., 4:8], outputs[24])
    torch.testing.assert_close(features[..., 8:12], outputs[16])
    torch.testing.assert_close(features[..., 12:], outputs[8])
    collector.close()


def test_td_basis_fold_rejects_raw_calibration_and_double_fold():
    raw = TargetFeatEmbedWarp(
        _ToyDraft(), 16, 0.02, False,
        {"raw_scale": torch.ones(16), "raw_bias": torch.zeros(16)})
    with pytest.raises(ValueError, match="raw feature calibration"):
        raw.fold_target_basis_(torch.ones(4), torch.ones(4))

    wrapper = TargetFeatEmbedWarp(_ToyDraft(), 16, 0.02, False)
    wrapper.fold_target_basis_(torch.ones(4), torch.ones(4))
    assert wrapper.td_basis_folded
    with pytest.raises(RuntimeError, match="already folded"):
        wrapper.fold_target_basis_(torch.ones(4), torch.ones(4))


@pytest.mark.parametrize("bad_weight", [
    torch.randn(2, 15),
    torch.randn(2, 4, 4),
])
def test_td_basis_fold_validates_projection_shape(bad_weight):
    with pytest.raises(ValueError, match="projection weight"):
        fold_td_projection_weight(
            bad_weight, torch.ones(4), torch.ones(4), 4)


@pytest.mark.parametrize("make_parser", [generation_parser, benchmark_parser])
def test_td_basis_fold_cli_is_opt_in(make_parser):
    argv = ["--mode", "pard2-td"]
    if make_parser is benchmark_parser:
        argv += ["--dataset", "humaneval", "--output", "unused.json"]
    assert make_parser().parse_args(argv).td_basis_fold is False
    assert make_parser().parse_args(argv + ["--td-basis-fold"]).td_basis_fold is True
