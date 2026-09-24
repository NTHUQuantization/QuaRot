"""CPU-only contracts for the canonical Qwen3-14B PARD2 profile."""

import json
from types import SimpleNamespace

import pytest

from e2e.benchmark_pard2 import (
    QWEN3_14B_ARCHITECTURE,
    QWEN3_14B_SOURCE_CONFIG_SHA256,
    QWEN3_14B_SOURCE_INDEX_SHA256,
    QWEN3_14B_SOURCE_MODEL_ID,
    QWEN3_14B_SOURCE_REVISION,
    _qwen3_32b_expected_weight_keys,
    target_profile_preflight,
)
from e2e.qualify_pard2 import (
    PROMPT_COUNTS,
    validate_formal_payload,
)
from e2e.speculative import Pard2Spec


DRAFT_8B_REVISION = "67a1516c8f6fc145cda99916799a0cbb3a4af135"
DRAFT_14B_REVISION = "679eff0b65ffaf5abd2dadd21a17909562935798"


def _draft_config(target_dim=20480):
    return SimpleNamespace(
        model_type="qwen3",
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        vocab_size=151936,
        pard_token=151670,
        pard2_target_dim=target_dim,
        pard2_target_layers=[-1, -8, -16, -24],
        pard2_scale=0.02,
        pard2_proj_bias=False,
    )


def _target_config(revision=QWEN3_14B_SOURCE_REVISION):
    return SimpleNamespace(
        model_type="qwen3_quarot",
        **QWEN3_14B_ARCHITECTURE,
        tokenizer_name_or_path=(
            "/cache/models--Qwen--Qwen3-14B/snapshots/" + revision),
    )


def _packed_target_config(**overrides):
    config = {
        **QWEN3_14B_ARCHITECTURE,
        "model_type": "qwen3_quarot",
        "quarot_source_model_id": QWEN3_14B_SOURCE_MODEL_ID,
        "quarot_source_revision": QWEN3_14B_SOURCE_REVISION,
        "quarot_source_config_sha256": QWEN3_14B_SOURCE_CONFIG_SHA256,
        "quarot_source_index_sha256": QWEN3_14B_SOURCE_INDEX_SHA256,
        "quarot_checkpoint_format_version": 2,
        "quarot_ffn_format": "grouped_h256_v1",
        "quarot_activation_clip_ratio": 0.9,
        "quarot_conversion": {
            "version": 2,
            "method": "rtn",
            "seed": 0,
            "rotation_device": "cuda",
            "rotation_dtype": "float32",
            "w_bits": 4,
            "w_groupsize": -1,
            "w_asym": False,
            "w_clip": True,
        },
        "quarot_rotation_format": "hadk_v1",
        "quarot_rotation_width": 5120,
        "quarot_rotation_remainder": 40,
        "quarot_rotation_inner": 128,
        "quarot_rotation_seed": 0,
        "quarot_rotation_device": "cuda",
        "quarot_rotation_dtype": "float32",
        "quarot_rotation_signs": [1, -1] * 2560,
        "quarot_final_norm_weight": [1.0] * 5120,
    }
    config.update(overrides)
    return config


def _write_packed_target(target, **overrides):
    target.mkdir()
    config = _packed_target_config(**overrides)
    (target / "config.json").write_text(json.dumps(config))
    weight_map = {}
    for key in _qwen3_32b_expected_weight_keys(layer_count=40):
        if key.startswith("model.layers."):
            layer = int(key.split(".")[2])
            filename = f"model-layer-{layer:05d}.safetensors"
        else:
            filename = "model-global.safetensors"
        weight_map[key] = filename
    for filename in set(weight_map.values()):
        (target / filename).write_bytes(b"x")
    (target / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": weight_map,
    }))
    return config


def test_qwen3_14b_td_profile_resolves_pinned_models_and_projection_width():
    spec = Pard2Spec.for_benchmark_profile("qwen3_14b", mode="pard2-td")

    assert spec.model_id == "amd/PARD2-Qwen3-14B"
    assert spec.revision == DRAFT_14B_REVISION
    assert spec.target_model_id == QWEN3_14B_SOURCE_MODEL_ID
    assert spec.target_revision == QWEN3_14B_SOURCE_REVISION
    assert spec.target_dim == 20480
    assert spec.benchmark_profile == "qwen3_14b"
    assert spec.target_alignment == "strict"

    with pytest.raises(ValueError, match="unknown PARD2 benchmark profile"):
        Pard2Spec.for_benchmark_profile("qwen3_14")


def test_qwen3_14b_ti_profile_uses_shared_qwen3_family_drafter():
    spec = Pard2Spec.for_benchmark_profile("qwen3_14b", mode="pard2-ti")

    assert spec.model_id == "amd/PARD2-Qwen3-8B"
    assert spec.revision == DRAFT_8B_REVISION
    assert spec.target_model_id == QWEN3_14B_SOURCE_MODEL_ID
    assert spec.target_revision == QWEN3_14B_SOURCE_REVISION
    assert spec.target_dim == 16384
    assert spec.benchmark_profile == "qwen3_14b"


def test_qwen3_14b_profile_enforces_40_layer_target_architecture():
    spec = Pard2Spec.for_benchmark_profile("qwen3_14b", mode="pard2-ti")
    pinned_draft = "/cache/snapshots/" + DRAFT_8B_REVISION
    spec.validate(
        _target_config(), _draft_config(16384), mode="pard2-ti",
        draft_snapshot=pinned_draft)

    wrong = _target_config()
    wrong.num_hidden_layers = 64
    with pytest.raises(ValueError, match="num_hidden_layers must be 40"):
        spec.validate(
            wrong, _draft_config(16384), mode="pard2-ti",
            draft_snapshot=pinned_draft)


@pytest.mark.parametrize("mode", ("pard2-ti", "pard2-td"))
def test_qwen3_14b_rejects_wrong_draft_revision(mode):
    spec = Pard2Spec.for_benchmark_profile("qwen3_14b", mode=mode)
    target_dim = 16384 if mode == "pard2-ti" else 20480
    with pytest.raises(ValueError, match="pinned PARD2-Qwen3"):
        spec.validate(
            _target_config(),
            _draft_config(target_dim),
            mode=mode,
            draft_snapshot="/cache/snapshots/" + "0" * 40,
        )


@pytest.mark.parametrize("mode", ("pard2-ti", "pard2-td"))
def test_qwen3_14b_rejects_wrong_target_revision(mode):
    spec = Pard2Spec.for_benchmark_profile("qwen3_14b", mode=mode)
    draft_revision = (
        DRAFT_8B_REVISION if mode == "pard2-ti" else DRAFT_14B_REVISION)
    target_dim = 16384 if mode == "pard2-ti" else 20480
    with pytest.raises(ValueError, match="target converted from the pinned"):
        spec.validate(
            _target_config(revision="0" * 40),
            _draft_config(target_dim),
            mode=mode,
            draft_snapshot="/cache/snapshots/" + draft_revision,
        )


def test_qwen3_14b_cpu_preflight_accepts_40_layer_packed_provenance(tmp_path):
    target = tmp_path / "converted"
    _write_packed_target(target)

    provenance = target_profile_preflight(target, "qwen3_14b")

    assert provenance["source_model_id"] == QWEN3_14B_SOURCE_MODEL_ID
    assert provenance["source_revision"] == QWEN3_14B_SOURCE_REVISION
    assert provenance["source_config"] == QWEN3_14B_ARCHITECTURE
    checkpoint = provenance["checkpoint_contract"]
    assert checkpoint["quant_method"] == "rtn"
    assert checkpoint["target_weight_map_entries"] == 642
    assert checkpoint["target_shard_count"] == 41


def test_qwen3_14b_cpu_preflight_rejects_wrong_source_revision(tmp_path):
    target = tmp_path / "converted"
    _write_packed_target(target, quarot_source_revision="0" * 40)

    with pytest.raises(ValueError, match="target source provenance mismatch"):
        target_profile_preflight(target, "qwen3_14b")


def test_qwen3_14b_qualifier_consumes_strict_target_provenance(tmp_path):
    target = tmp_path / "converted"
    _write_packed_target(target)
    provenance = target_profile_preflight(target, "qwen3_14b")
    dataset = "math_500"
    contract = {
        "mode": "pard2-td",
        "dataset": dataset,
        "formal_protocol": True,
        "qualified": True,
        "prompt_count": PROMPT_COUNTS[dataset],
        "generated_tokens": 256,
        "batch_size": 1,
        "warmups": 8,
        "sweeps": 3,
        "greedy": True,
        "ignore_eos": False,
        "benchmark_profile": "qwen3_14b",
        "qualification_track": "canonical",
        "target_alignment": "strict",
        "td_proxy_profile": None,
        "target_model_id": QWEN3_14B_SOURCE_MODEL_ID,
        "target_revision": QWEN3_14B_SOURCE_REVISION,
        "target_provenance": provenance,
    }
    payload = {
        "contract": contract,
        "runs": [{} for _ in range(PROMPT_COUNTS[dataset] * 3)],
    }

    validate_formal_payload(
        payload,
        "pard2-td",
        dataset,
        benchmark_profile="qwen3_14b",
        qualification_track="canonical",
        target_alignment="strict",
        td_proxy_profile=None,
    )

    contract["target_revision"] = "0" * 40
    with pytest.raises(ValueError, match="target_revision"):
        validate_formal_payload(
            payload,
            "pard2-td",
            dataset,
            benchmark_profile="qwen3_14b",
            qualification_track="canonical",
            target_alignment="strict",
            td_proxy_profile=None,
        )
