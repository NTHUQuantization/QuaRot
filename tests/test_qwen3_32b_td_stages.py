import json

import torch

from e2e.qwen3_32b_td_features import (
    SufficientStats,
    TAP_CANDIDATES,
    UNION_TAPS,
    selected_samples,
)


def test_streamed_affine_recovers_exact_channel_map():
    generator = torch.Generator().manual_seed(29)
    source = torch.randn(37, 11, generator=generator)
    wanted_scale = torch.linspace(0.5, 2.0, 11)
    wanted_bias = torch.linspace(-1.0, 1.0, 11)
    reference = source * wanted_scale + wanted_bias
    stats = SufficientStats(11)
    stats.update(source[:13], reference[:13])
    stats.update(source[13:], reference[13:])

    scale, bias, metrics = stats.fit()

    torch.testing.assert_close(scale, wanted_scale, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(bias, wanted_bias, atol=1e-5, rtol=1e-5)
    assert metrics["tokens"] == 37
    assert metrics["rmse_after"] < 1e-3
    assert metrics["cosine_after"] > 0.999999


def test_tap_candidates_are_four_wide_and_union_preserves_final_first():
    assert UNION_TAPS[0] == -1
    assert all(len(taps) == 4 for taps in TAP_CANDIDATES.values())
    assert all(tap in UNION_TAPS for taps in TAP_CANDIDATES.values() for tap in taps)


def test_selected_samples_truncates_exactly_to_budget(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        {"sample_id": "a", "input_ids": list(range(10))},
        {"sample_id": "b", "input_ids": list(range(20))},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    selected = list(selected_samples(path, 25))

    assert [row[0]["sample_id"] for row in selected] == ["a", "b"]
    assert [len(row[1]) for row in selected] == [10, 15]
