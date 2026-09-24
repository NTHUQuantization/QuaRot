import json

import pytest
import torch

from e2e.benchmark_pard2 import (
    QWEN3_32B_ARCHITECTURE,
    QWEN3_32B_SOURCE_CONFIG_SHA256,
    QWEN3_32B_SOURCE_INDEX_SHA256,
    QWEN3_32B_SOURCE_MODEL_ID,
    QWEN3_32B_SOURCE_REVISION,
    _qwen3_32b_expected_weight_keys,
    main as benchmark_main,
    parser as benchmark_parser,
    target_profile_preflight,
    token_ids_sha256,
)
from e2e.qualify_pard2 import (
    DATASETS,
    PROMPT_COUNTS,
    QWEN3_32B_CHECKPOINT_CONTRACT,
    evaluate_mode,
    load_payloads,
    main as qualify_main,
    memory_peaks,
    validate_formal_payload,
)


PROXY = "qwen3-14b-on-qwen3-32b"


def formal_contract(mode, dataset, *, profile="qwen3_32b",
                    track="canonical", alignment="strict", proxy=None):
    contract = {
        "mode": mode,
        "dataset": dataset,
        "formal_protocol": True,
        "qualified": track == "canonical",
        "prompt_count": PROMPT_COUNTS[dataset],
        "generated_tokens": 256,
        "batch_size": 1,
        "warmups": 8,
        "sweeps": 3,
        "greedy": True,
        "ignore_eos": False,
        "benchmark_profile": profile,
        "qualification_track": track,
        "target_alignment": alignment,
        "td_proxy_profile": proxy,
    }
    if profile == "qwen3_32b":
        provenance = {
            "source_model_id": QWEN3_32B_SOURCE_MODEL_ID,
            "source_revision": QWEN3_32B_SOURCE_REVISION,
            "source_config_sha256": QWEN3_32B_SOURCE_CONFIG_SHA256,
            "source_index_sha256": QWEN3_32B_SOURCE_INDEX_SHA256,
            "source_config": dict(QWEN3_32B_ARCHITECTURE),
            "target_model_type": "qwen3_quarot",
            "target_config_sha256": "a" * 64,
            "checkpoint_contract": {
                **QWEN3_32B_CHECKPOINT_CONTRACT,
                "target_index_sha256": "c" * 64,
            },
        }
        contract.update({
            "target_model_id": provenance["source_model_id"],
            "target_revision": provenance["source_revision"],
            "target_provenance": provenance,
        })
    return contract


def qwen3_32b_target_config(**overrides):
    config = {
        **QWEN3_32B_ARCHITECTURE,
        "model_type": "qwen3_quarot",
        "quarot_source_model_id": QWEN3_32B_SOURCE_MODEL_ID,
        "quarot_source_revision": QWEN3_32B_SOURCE_REVISION,
        "quarot_source_config_sha256": QWEN3_32B_SOURCE_CONFIG_SHA256,
        "quarot_source_index_sha256": QWEN3_32B_SOURCE_INDEX_SHA256,
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


def qwen3_32b_gptq_conversion(**overrides):
    conversion = {
        "version": 5,
        "method": "gptq",
        "dataset": "wikitext2",
        "nsamples": 128,
        "seqlen": 2048,
        "percdamp": 0.01,
        "act_order": False,
        "seed": 0,
        "rotation_device": "cuda",
        "rotation_dtype": "float32",
        "w_bits": 4,
        "w_groupsize": -1,
        "w_asym": False,
        "w_clip": True,
    }
    conversion.update(overrides)
    return conversion


def write_qwen3_32b_target(target, **overrides):
    target.mkdir()
    config = qwen3_32b_target_config(**overrides)
    (target / "config.json").write_text(json.dumps(config))
    weight_map = {}
    for key in _qwen3_32b_expected_weight_keys():
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


def test_qwen3_32b_cpu_preflight_accepts_exact_path_independent_identity(
        tmp_path):
    target = tmp_path / "converted"
    write_qwen3_32b_target(target)
    provenance = target_profile_preflight(target, "qwen3_32b")

    assert provenance["source_model_id"] == QWEN3_32B_SOURCE_MODEL_ID
    assert provenance["source_revision"] == QWEN3_32B_SOURCE_REVISION
    assert provenance[
        "source_config_sha256"] == QWEN3_32B_SOURCE_CONFIG_SHA256
    assert provenance[
        "source_index_sha256"] == QWEN3_32B_SOURCE_INDEX_SHA256
    assert provenance["source_config"] == QWEN3_32B_ARCHITECTURE
    assert len(provenance["target_config_sha256"]) == 64
    checkpoint = provenance["checkpoint_contract"]
    assert checkpoint["target_weight_map_entries"] == 1026
    assert checkpoint["target_shard_count"] == 65
    assert len(checkpoint["target_index_sha256"]) == 64
    assert str(target) not in json.dumps(provenance)


def test_qwen3_32b_cpu_preflight_accepts_exact_gptq_contract(tmp_path):
    target = tmp_path / "converted"
    write_qwen3_32b_target(
        target, quarot_conversion=qwen3_32b_gptq_conversion())
    provenance = target_profile_preflight(target, "qwen3_32b")

    assert provenance["checkpoint_contract"]["quant_method"] == "gptq"


@pytest.mark.parametrize("field,value", (
    ("version", 2),
    ("dataset", "c4"),
    ("nsamples", 127),
    ("seqlen", 1024),
    ("percdamp", 0.02),
    ("act_order", True),
))
def test_qwen3_32b_cpu_preflight_rejects_wrong_gptq_contract(
        tmp_path, field, value):
    target = tmp_path / "converted"
    write_qwen3_32b_target(
        target,
        quarot_conversion=qwen3_32b_gptq_conversion(**{field: value}))

    with pytest.raises(ValueError, match=f"quarot_conversion.{field}"):
        target_profile_preflight(target, "qwen3_32b")


def test_qwen3_32b_cpu_preflight_rejects_8b_labeled_as_32b(tmp_path):
    target = tmp_path / "converted"
    write_qwen3_32b_target(
        target,
        hidden_size=4096, intermediate_size=12288,
        num_hidden_layers=36, num_attention_heads=32)

    with pytest.raises(ValueError, match="target architecture mismatch"):
        target_profile_preflight(target, "qwen3_32b")


def test_qwen3_32b_cpu_preflight_rejects_dense_checkpoint(tmp_path):
    target = tmp_path / "converted"
    write_qwen3_32b_target(target, model_type="qwen3")
    with pytest.raises(ValueError, match="W4A4KV4 checkpoint contract"):
        target_profile_preflight(target, "qwen3_32b")



@pytest.mark.parametrize(
    "field", ("quarot_rotation_signs", "quarot_final_norm_weight"))
def test_qwen3_32b_cpu_preflight_rejects_boolean_metadata(tmp_path, field):
    target = tmp_path / "converted"
    value = ([True, -1] * 2560 if field == "quarot_rotation_signs"
             else [True] * 5120)
    write_qwen3_32b_target(target, **{field: value})
    with pytest.raises(ValueError, match=field):
        target_profile_preflight(target, "qwen3_32b")

def test_qwen3_32b_cpu_preflight_rejects_misplaced_packed_weight(tmp_path):
    target = tmp_path / "converted"
    write_qwen3_32b_target(target)
    index_path = target / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    key = "model.layers.0.self_attn.q_proj.weight"
    index["weight_map"][key] = "model-global.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="weight map mismatch"):
        target_profile_preflight(target, "qwen3_32b")


def test_qwen3_32b_formal_contract_rejects_wrong_source_config():
    dataset = "math_500"
    contract = formal_contract("pard2-ti", dataset)
    contract["target_provenance"]["source_config"]["hidden_size"] = 4096
    payload = {
        "contract": contract,
        "runs": [{} for _ in range(PROMPT_COUNTS[dataset] * 3)],
    }

    with pytest.raises(ValueError, match="target_provenance.source_config"):
        validate_formal_payload(
            payload, "pard2-ti", dataset, benchmark_profile="qwen3_32b",
            qualification_track="canonical", target_alignment="strict",
            td_proxy_profile=None)


def test_benchmark_defaults_preserve_formal_normal_eos_contract():
    args = benchmark_parser().parse_args([
        "--mode", "ar", "--dataset", "humaneval", "--output", "unused.json",
    ])
    assert args.generated_tokens == 256
    assert args.max_cache_len == 8192
    assert args.warmups == 8
    assert args.sweeps == 3
    assert args.ignore_eos is False
    assert args.benchmark_profile == "qwen3_8b"
    assert args.td_proxy_profile is None
    assert args.expected_hip_sha256 is None


def test_benchmark_max_cache_len_override_and_pre_gpu_validation():
    expected_hip_sha256 = "0" * 64
    args = benchmark_parser().parse_args([
        "--mode", "pard2-ti", "--dataset", "humaneval",
        "--max-cache-len", "2048",
        "--expected-hip-sha256", expected_hip_sha256,
        "--output", "unused.json",
    ])
    assert args.max_cache_len == 2048
    assert args.expected_hip_sha256 == expected_hip_sha256

    with pytest.raises(ValueError, match="--max-cache-len must be positive"):
        benchmark_main([
            "--mode", "ar", "--dataset", "humaneval",
            "--max-cache-len", "0", "--output", "unused.json",
        ])
    with pytest.raises(ValueError, match="expected_hip_sha256"):
        benchmark_main([
            "--mode", "ar", "--dataset", "humaneval",
            "--expected-hip-sha256", "not-a-sha",
            "--output", "unused.json",
        ])


def test_token_ids_sha256_is_device_and_integer_dtype_independent():
    input_ids = torch.tensor([[1, 2, 151935]], dtype=torch.int64)
    digest = token_ids_sha256(input_ids)
    assert len(digest) == 64
    assert digest == token_ids_sha256(input_ids.to(dtype=torch.int32))
    assert digest != token_ids_sha256(torch.tensor([[1, 2, 151934]]))


def test_cross_target_proxy_is_narrowly_opted_in_before_gpu_preflight():
    common = [
        "--dataset", "humaneval", "--output", "unused.json",
        "--td-proxy-profile", PROXY,
    ]
    with pytest.raises(ValueError, match="valid only for pard2-td"):
        benchmark_main(["--mode", "ar", "--benchmark-profile", "qwen3_32b",
                        *common])
    with pytest.raises(ValueError, match="requires --benchmark-profile qwen3_32b"):
        benchmark_main(["--mode", "pard2-td", *common])


def test_formal_profile_metadata_rejects_proxy_as_canonical():
    dataset = "math_500"
    payload = {
        "contract": formal_contract("pard2-ti", dataset),
        "runs": [{} for _ in range(PROMPT_COUNTS[dataset] * 3)],
    }
    validate_formal_payload(
        payload, "pard2-ti", dataset, benchmark_profile="qwen3_32b",
        qualification_track="canonical", target_alignment="strict",
        td_proxy_profile=None)
    payload["contract"].update({
        "qualification_track": "experimental",
        "target_alignment": "cross_target_proxy",
        "td_proxy_profile": PROXY,
    })
    with pytest.raises(ValueError, match="not a formal benchmark payload"):
        validate_formal_payload(
            payload, "pard2-ti", dataset, benchmark_profile="qwen3_32b",
            qualification_track="canonical", target_alignment="strict",
            td_proxy_profile=None)


def test_qwen3_32b_qualifier_accepts_rtn_and_gptq_only():
    dataset = "math_500"
    for method in ("rtn", "gptq"):
        contract = formal_contract("ar", dataset)
        contract["target_provenance"]["checkpoint_contract"][
            "quant_method"] = method
        validate_formal_payload(
            {"contract": contract,
             "runs": [{} for _ in range(PROMPT_COUNTS[dataset] * 3)]},
            "ar", dataset, benchmark_profile="qwen3_32b",
            qualification_track="canonical", target_alignment="strict",
            td_proxy_profile=None)

    contract["target_provenance"]["checkpoint_contract"][
        "quant_method"] = "unknown"
    with pytest.raises(ValueError, match="quant_method"):
        validate_formal_payload(
            {"contract": contract,
             "runs": [{} for _ in range(PROMPT_COUNTS[dataset] * 3)]},
            "ar", dataset, benchmark_profile="qwen3_32b",
            qualification_track="canonical", target_alignment="strict",
            td_proxy_profile=None)


def test_legacy_qwen3_8b_profile_metadata_defaults_remain_valid():
    dataset = "math_500"
    contract = formal_contract(
        "ar", dataset, profile="qwen3_8b")
    for key in (
            "benchmark_profile", "qualification_track", "target_alignment",
            "td_proxy_profile", "formal_protocol"):
        contract.pop(key)
    payload = {
        "contract": contract,
        "runs": [{} for _ in range(PROMPT_COUNTS[dataset] * 3)],
    }
    validate_formal_payload(
        payload, "ar", dataset, benchmark_profile="qwen3_8b",
        qualification_track="canonical", target_alignment="strict",
        td_proxy_profile=None)


def test_memory_peaks_preserve_allocated_key_and_gate_on_largest_counter():
    payload = {
        "gpu_preflight": {"total_bytes": 1000},
        "memory_snapshots": [{"device_used_bytes": 130}],
        "external_vram_monitor": {"peak_device_used_bytes": 140},
        "runs": [{
            "peak_vram_bytes": 100,
            "peak_vram_allocated_bytes": 100,
            "peak_vram_reserved_bytes": 120,
            "memory_snapshot": {"device_used_bytes": 125},
        }],
    }
    peaks = memory_peaks(payload)
    assert peaks["peak_vram_bytes"] == 100
    assert peaks["peak_vram_allocated_bytes"] == 100
    assert peaks["peak_vram_reserved_bytes"] == 120
    assert peaks["peak_device_used_bytes"] == 140
    assert peaks["vram_gate_peak_bytes"] == 140
    assert peaks["vram_headroom"] == pytest.approx(0.86)


def test_qualifier_vram_gate_fraction_is_configurable(monkeypatch):
    rows = [{
        "sweep": sweep, "prompt_index": 0, "output_ids": [1],
        "steady_tokens_per_s": 10.0, "mean_accept_length": 2.0,
        "peak_vram_bytes": 94, "peak_vram_reserved_bytes": 94,
    } for sweep in range(3)]
    ar = {"gpu_preflight": {"total_bytes": 100}, "runs": [
        {**row, "steady_tokens_per_s": 5.0} for row in rows]}
    current = {"gpu_preflight": {"total_bytes": 100}, "runs": rows}
    monkeypatch.setattr(
        "e2e.qualify_pard2.bootstrap_ci", lambda values: (2.0, 2.0))
    strict = evaluate_mode(
        ar, current, "pard2-ti", "humaneval", 1, False,
        vram_gate_fraction=0.90)
    relaxed = evaluate_mode(
        ar, current, "pard2-ti", "humaneval", 1, False,
        vram_gate_fraction=0.95)
    assert strict["hard_gate"] is False
    assert relaxed["hard_gate"] is True
    assert relaxed["vram_gate_fraction"] == 0.95


def write_payload(root, mode, dataset, speed, *, experimental=False):
    rows = []
    for sweep in range(3):
        for prompt in range(PROMPT_COUNTS[dataset]):
            rows.append({
                "sweep": sweep,
                "prompt_index": prompt,
                "output_ids": [prompt, 7],
                "steady_tokens_per_s": speed,
                "mean_accept_length": 2.0,
                "peak_vram_bytes": 100,
                "peak_vram_reserved_bytes": 120,
            })
    payload = {
        "contract": formal_contract(
            mode, dataset,
            track="experimental" if experimental else "canonical",
            alignment="cross_target_proxy" if experimental else "strict",
            proxy=PROXY if experimental else None),
        "gpu_preflight": {"total_bytes": 1000},
        "runs": rows,
    }
    (root / f"{mode}_{dataset}.json").write_text(json.dumps(payload))


def test_qualifier_loads_runner_safe_ti_filenames(tmp_path):
    for dataset in DATASETS:
        write_payload(tmp_path, "ar", dataset, 10.0)
        write_payload(tmp_path, "pard2-ti", dataset, 20.0)
        (tmp_path / f"pard2-ti_{dataset}.json").rename(
            tmp_path / f"pard2_ti_{dataset}.json")

    payloads = load_payloads(tmp_path, ("ar", "pard2-ti"))
    assert len(payloads) == 6
    assert payloads[("pard2-ti", "humaneval")]["contract"]["mode"] == "pard2-ti"


def test_qwen3_32b_qualifier_rejects_mixed_target_artifacts(tmp_path):
    for dataset in DATASETS:
        write_payload(tmp_path, "ar", dataset, 10.0)
        write_payload(tmp_path, "pard2-ti", dataset, 20.0)
    changed_path = tmp_path / "pard2-ti_math_500.json"
    changed = json.loads(changed_path.read_text())
    changed["contract"]["target_provenance"][
        "target_config_sha256"] = "b" * 64
    changed_path.write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="different target artifacts"):
        qualify_main([
            "--result-dir", str(tmp_path),
            "--benchmark-profile", "qwen3_32b",
            "--output", str(tmp_path / "qualification.json"),
        ])


def test_qwen3_32b_proxy_gate_is_isolated_from_canonical(monkeypatch, tmp_path):
    for dataset in DATASETS:
        write_payload(tmp_path, "ar", dataset, 10.0)
        write_payload(tmp_path, "pard2-ti", dataset, 20.0)
        write_payload(
            tmp_path, "pard2-td", dataset, 5.0, experimental=True)
    monkeypatch.setattr(
        "e2e.qualify_pard2.bootstrap_ci",
        lambda values: (min(values), max(values)))
    output = tmp_path / "qualification.json"
    qualify_main([
        "--result-dir", str(tmp_path),
        "--benchmark-profile", "qwen3_32b",
        "--td-proxy-profile", PROXY,
        "--output", str(output),
    ])
    report = json.loads(output.read_text())
    assert report["hard_gate"] is True
    assert report["all_exact_parity"] is True
    assert report["canonical_modes"] == ["ar", "pard2-ti"]
    assert report["target_provenance"]["source_model_id"] == QWEN3_32B_SOURCE_MODEL_ID
    assert report["experimental"]["included"] is True
    assert report["experimental"]["all_exact_parity"] is True
    assert report["experimental"]["hard_gate"] is False
    assert report["td_stretch_gate"] is None
