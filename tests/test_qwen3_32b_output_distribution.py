from types import SimpleNamespace

import json
import torch

from e2e.audit_qwen3_32b_checkpoint import (
    _expected_conversion_signature)
from e2e.compare_qwen3_32b_output_distribution import (
    _decoder_hidden, _metrics, compare)


def test_decoder_hidden_preserves_tensor_and_unwraps_tuple():
    hidden = torch.randn(1, 3, 4)

    assert _decoder_hidden(hidden) is hidden
    assert _decoder_hidden((hidden, "cache")) is hidden


def test_gptq_audit_signature_is_exact():
    signature = _expected_conversion_signature({"model": "source"}, "gptq")
    assert signature == {
        "version": 5,
        "method": "gptq",
        "model": "source",
        "seed": 0,
        "rotation_device": "cuda",
        "rotation_dtype": "float32",
        "w_bits": 4,
        "w_groupsize": -1,
        "w_asym": False,
        "w_clip": True,
        "percdamp": 0.01,
        "act_order": False,
        "dataset": "wikitext2",
        "nsamples": 128,
        "seqlen": 2048,
    }


def test_identical_distribution_metrics_are_exact():
    logits = torch.tensor([1.0, 2.0, 3.0])
    metrics = _metrics(logits, logits)
    assert metrics["logit_rmse"] == 0
    assert metrics["logit_mae"] == 0
    assert metrics["kl_reference_to_candidate"] == 0
    assert metrics["total_variation"] == 0
    assert metrics["logit_cosine"] == 1
    assert metrics["top1_match"]
    assert metrics["top5_overlap"] == 3


def test_compare_reports_when_gptq_is_closer(tmp_path):
    records = [{
        "dataset": "humaneval",
        "offset": 0,
        "input_ids_sha256": "a" * 64,
        "input_token_count": 3,
    }]
    reference = {
        "kind": "bf16_reference",
        "records": records,
        "logits": [torch.tensor([[0.0, 1.0, 2.0]])],
    }
    rtn = {
        "kind": "quarot_target",
        "conversion": {"method": "rtn"},
        "records": records,
        "logits": [torch.tensor([[2.0, 1.0, 0.0]])],
    }
    gptq = {
        "kind": "quarot_target",
        "conversion": {"method": "gptq"},
        "records": records,
        "logits": [torch.tensor([[0.0, 1.1, 1.9]])],
    }
    paths = {
        "reference": tmp_path / "reference.pt",
        "rtn": tmp_path / "rtn.pt",
        "gptq": tmp_path / "gptq.pt",
        "output": tmp_path / "comparison.json",
    }
    torch.save(reference, paths["reference"])
    torch.save(rtn, paths["rtn"])
    torch.save(gptq, paths["gptq"])

    compare(SimpleNamespace(**paths))

    result = json.loads(paths["output"].read_text())
    assert all(result["gptq_lower_gap_than_rtn"].values())
    assert result["aggregates"]["gptq"]["top1_match_rate"] == 1
    assert result["aggregates"]["rtn"]["top1_match_rate"] == 0
