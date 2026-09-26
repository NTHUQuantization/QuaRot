"""Apply parity, speed, stability, VRAM and literature gates to benchmark JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics


DATASETS = ("humaneval", "gsm8k", "math_500")
MODES = ("ar", "pard2-ti", "pard2-td")
PROMPT_COUNTS = {"humaneval": 80, "gsm8k": 80, "math_500": 20}
PROFILE_MODES = {
    "qwen3_8b": MODES,
    "qwen3_14b": MODES,
    "qwen3_32b": ("ar", "pard2-ti"),
}
TD_PROXY_PROFILES = ("qwen3-14b-on-qwen3-32b",)
QWEN3_14B_SOURCE_IDENTITY = {
    "source_model_id": "Qwen/Qwen3-14B",
    "source_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
    "source_config_sha256": (
        "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a"),
    "source_index_sha256": (
        "62d7ad35757bae5e7baa452cb1483178b7daa50e869e923226b8da10871f7ebc"),
}
QWEN3_14B_SOURCE_CONFIG = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 40,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}
QWEN3_32B_SOURCE_IDENTITY = {
    "source_model_id": "Qwen/Qwen3-32B",
    "source_revision": "9216db5781bf21249d130ec9da846c4624c16137",
    "source_config_sha256": (
        "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb"),
    "source_index_sha256": (
        "bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0"),
}
QWEN3_32B_SOURCE_CONFIG = {
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_hidden_layers": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
}
QWEN3_32B_CHECKPOINT_CONTRACT = {
    "checkpoint_format_version": 2,
    "ffn_format": "grouped_h256_v1",
    "activation_clip_ratio": 0.9,
    "quant_method": "rtn",
    "weight_bits": 4,
    "activation_bits": 4,
    "kv_bits": 4,
    "weight_groupsize": -1,
    "weight_symmetric": True,
    "weight_clip": True,
    "rotation_format": "hadk_v1",
    "rotation_remainder": 40,
    "rotation_inner": 128,
    "rotation_sign_count": 5120,
    "final_norm_count": 5120,
    "target_weight_map_entries": 1026,
    "target_shard_count": 65,
}
QWEN3_14B_CHECKPOINT_CONTRACT = {
    **QWEN3_32B_CHECKPOINT_CONTRACT,
    "target_weight_map_entries": 642,
    "target_shard_count": 41,
}
STRICT_PROFILE_SOURCE_IDENTITIES = {
    "qwen3_14b": QWEN3_14B_SOURCE_IDENTITY,
    "qwen3_32b": QWEN3_32B_SOURCE_IDENTITY,
}
STRICT_PROFILE_SOURCE_CONFIGS = {
    "qwen3_14b": QWEN3_14B_SOURCE_CONFIG,
    "qwen3_32b": QWEN3_32B_SOURCE_CONFIG,
}
STRICT_PROFILE_CHECKPOINT_CONTRACTS = {
    "qwen3_14b": QWEN3_14B_CHECKPOINT_CONTRACT,
    "qwen3_32b": QWEN3_32B_CHECKPOINT_CONTRACT,
}
QWEN3_32B_QUANT_METHODS = ("rtn", "gptq")
LITERATURE = {
    # Official AMD vLLM-v1 Qwen3-8B table. It does not report Qwen MATH-500
    # or acceptance length, so those literature fractions are not invented.
    "humaneval": {"speedup": 6.75},
    "gsm8k": {"speedup": 6.44},
}


def keyed(payload):
    return {(row["sweep"], row["prompt_index"]): row for row in payload["runs"]}


def bootstrap_ci(values, samples=10_000, seed=0x50415244):
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.median(draw))
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def cv_by_sweep(rows, field):
    sweeps = sorted({row["sweep"] for row in rows})
    values = [statistics.median(row[field] for row in rows if row["sweep"] == sweep)
              for sweep in sweeps]
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else float("inf")


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _strict_profile_provenance_mismatches(contract, profile):
    source_identity = STRICT_PROFILE_SOURCE_IDENTITIES[profile]
    expected_source_config = STRICT_PROFILE_SOURCE_CONFIGS[profile]
    expected_checkpoint = STRICT_PROFILE_CHECKPOINT_CONTRACTS[profile]
    provenance = contract.get("target_provenance")
    if not isinstance(provenance, dict):
        return {"target_provenance": (provenance, "required mapping")}

    mismatches = {}
    for name, expected in source_identity.items():
        actual = provenance.get(name)
        if actual != expected:
            mismatches[f"target_provenance.{name}"] = (actual, expected)
    source_config = provenance.get("source_config")
    if not isinstance(source_config, dict):
        mismatches["target_provenance.source_config"] = (
            source_config, expected_source_config)
    else:
        for name, expected in expected_source_config.items():
            actual = source_config.get(name)
            if actual != expected:
                mismatches[
                    f"target_provenance.source_config.{name}"] = (
                        actual, expected)
    target_model_type = provenance.get("target_model_type")
    if target_model_type != "qwen3_quarot":
        mismatches["target_provenance.target_model_type"] = (
            target_model_type, "qwen3_quarot")
    target_config_sha256 = provenance.get("target_config_sha256")
    if not _valid_sha256(target_config_sha256):
        mismatches["target_provenance.target_config_sha256"] = (
            target_config_sha256, "lowercase SHA256")
    checkpoint = provenance.get("checkpoint_contract")
    if not isinstance(checkpoint, dict):
        mismatches["target_provenance.checkpoint_contract"] = (
            checkpoint, expected_checkpoint)
    else:
        for name, expected in expected_checkpoint.items():
            if name == "quant_method":
                continue
            actual = checkpoint.get(name)
            if actual != expected:
                mismatches[
                    f"target_provenance.checkpoint_contract.{name}"] = (
                        actual, expected)
        quant_method = checkpoint.get("quant_method")
        if quant_method not in QWEN3_32B_QUANT_METHODS:
            mismatches[
                "target_provenance.checkpoint_contract.quant_method"] = (
                    quant_method, QWEN3_32B_QUANT_METHODS)
        target_index_sha256 = checkpoint.get("target_index_sha256")
        if not _valid_sha256(target_index_sha256):
            mismatches[
                "target_provenance.checkpoint_contract.target_index_sha256"] = (
                    target_index_sha256, "lowercase SHA256")
    if contract.get("target_model_id") != source_identity["source_model_id"]:
        mismatches["target_model_id"] = (
            contract.get("target_model_id"),
            source_identity["source_model_id"])
    if contract.get("target_revision") != source_identity["source_revision"]:
        mismatches["target_revision"] = (
            contract.get("target_revision"),
            source_identity["source_revision"])
    return mismatches


def _qwen3_32b_provenance_mismatches(contract):
    return _strict_profile_provenance_mismatches(contract, "qwen3_32b")


def target_fingerprint(payload):
    """Return the path-independent identity used to compare benchmark files."""
    provenance = payload["contract"]["target_provenance"]
    checkpoint = provenance["checkpoint_contract"]
    return tuple(provenance.get(name) for name in (
        "source_model_id", "source_revision", "source_config_sha256",
        "source_index_sha256", "target_config_sha256")) + (
            checkpoint.get("target_index_sha256"),)


def validate_formal_payload(
        payload, mode, dataset, *, benchmark_profile=None,
        qualification_track=None, target_alignment=None,
        td_proxy_profile=None):
    contract = payload.get("contract", {})
    expected_qualified = qualification_track != "experimental"
    expected = {
        "mode": mode, "dataset": dataset,
        "qualified": expected_qualified,
        "prompt_count": PROMPT_COUNTS[dataset], "generated_tokens": 256,
        "batch_size": 1, "warmups": 8, "sweeps": 3, "greedy": True,
        "ignore_eos": False,
    }
    mismatches = {
        key: (contract.get(key), value) for key, value in expected.items()
        if contract.get(key) != value}
    formal_protocol = contract.get(
        "formal_protocol", contract.get("qualified"))
    if formal_protocol is not True:
        mismatches["formal_protocol"] = (formal_protocol, True)
    # Results produced before profiles were added are canonical Qwen3-8B
    # payloads. These defaults retain qualification compatibility for them.
    if benchmark_profile is not None:
        profiled = {
            "benchmark_profile": contract.get(
                "benchmark_profile", "qwen3_8b"),
            "qualification_track": contract.get(
                "qualification_track", "canonical"),
            "target_alignment": contract.get("target_alignment", "strict"),
            "td_proxy_profile": contract.get("td_proxy_profile"),
        }
        profile_expected = {
            "benchmark_profile": benchmark_profile,
            "qualification_track": qualification_track,
            "target_alignment": target_alignment,
            "td_proxy_profile": td_proxy_profile,
        }
        mismatches.update({
            key: (profiled[key], value)
            for key, value in profile_expected.items()
            if profiled[key] != value
        })
    if benchmark_profile in STRICT_PROFILE_SOURCE_IDENTITIES:
        mismatches.update(
            _strict_profile_provenance_mismatches(
                contract, benchmark_profile))
    expected_runs = PROMPT_COUNTS[dataset] * 3
    if len(payload.get("runs", ())) != expected_runs:
        mismatches["runs"] = (len(payload.get("runs", ())), expected_runs)
    if mismatches:
        raise ValueError(
            f"{mode}/{dataset} is not a formal benchmark payload: {mismatches}")


def memory_peaks(payload):
    """Normalize old allocated-only and new allocator/device memory schemas."""
    rows = payload["runs"]
    allocated = max(
        int(row.get("peak_vram_allocated_bytes",
                    row.get("peak_vram_bytes", 0)))
        for row in rows)
    reserved = max(
        int(row.get("peak_vram_reserved_bytes",
                    row.get("peak_vram_allocated_bytes",
                            row.get("peak_vram_bytes", 0))))
        for row in rows)
    snapshots = list(payload.get("memory_snapshots", ()))
    preflight_snapshot = payload.get("gpu_preflight", {}).get("memory_snapshot")
    if preflight_snapshot:
        snapshots.append(preflight_snapshot)
    snapshots.extend(
        row["memory_snapshot"] for row in rows if row.get("memory_snapshot"))
    device_used = max(
        (
            int(snapshot.get(
                "global_used_bytes", snapshot.get("device_used_bytes", 0)))
            for snapshot in snapshots
        ),
        default=allocated)
    external = payload.get("external_vram_monitor", {})
    device_used = max(
        device_used, int(external.get("peak_device_used_bytes", 0)))
    total = int(payload["gpu_preflight"]["total_bytes"])
    gate_peak = max(allocated, reserved, device_used)
    return {
        # Preserve the historical field as allocated bytes.
        "peak_vram_bytes": allocated,
        "peak_vram_allocated_bytes": allocated,
        "peak_vram_reserved_bytes": reserved,
        "peak_device_used_bytes": device_used,
        "peak_global_used_bytes": device_used,
        "vram_gate_peak_bytes": gate_peak,
        "vram_headroom": 1.0 - gate_peak / total,
        "vram_total_bytes": total,
    }


def evaluate_mode(ar_payload, current_payload, mode, dataset, phase,
                  literature_gate, vram_gate_fraction=0.90):
    ar = keyed(ar_payload)
    current = keyed(current_payload)
    same_keys = ar.keys() == current.keys()
    parity = same_keys and all(
        ar[key]["output_ids"] == current[key]["output_ids"] for key in ar)
    speedups = [
        current[key]["steady_tokens_per_s"] / ar[key]["steady_tokens_per_s"]
        for key in ar.keys() & current.keys()
        if ar[key]["steady_tokens_per_s"] > 0
    ]
    ci = bootstrap_ci(speedups)
    cv = cv_by_sweep(current_payload["runs"], "steady_tokens_per_s")
    memory = memory_peaks(current_payload)
    mean_accept = statistics.mean(
        row["mean_accept_length"] for row in current_payload["runs"])
    row = {
        "exact_parity": parity,
        "median_speedup": statistics.median(speedups) if speedups else 0.0,
        "speedup_bootstrap_95_ci": ci,
        "run_level_cv": cv,
        **memory,
        "mean_accept_length": mean_accept,
        "vram_gate_fraction": vram_gate_fraction,
    }
    row["hard_gate"] = (
        parity and ci[0] > 1.0 and cv < 0.05
        and memory["vram_gate_peak_bytes"]
        <= vram_gate_fraction * memory["vram_total_bytes"])
    if mode == "pard2-td" and literature_gate:
        reference = LITERATURE.get(dataset)
        if reference is not None:
            row["literature_speed_fraction"] = (
                row["median_speedup"] / reference["speedup"])
            threshold = 0.70 if phase == 1 else 0.80
            row["stretch_gate"] = (
                row["literature_speed_fraction"] >= threshold)
        else:
            row["stretch_gate"] = None
    return row


def result_path(root, mode, dataset):
    """Resolve runner-safe names while retaining legacy hyphenated results."""
    runner_path = root / f"{mode.replace('-', '_')}_{dataset}.json"
    legacy_path = root / f"{mode}_{dataset}.json"
    if runner_path.is_file() or runner_path == legacy_path:
        return runner_path
    return legacy_path


def load_payloads(root, modes):
    return {
        (mode, dataset): json.loads(
            result_path(root, mode, dataset).read_text())
        for mode in modes for dataset in DATASETS
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--benchmark-profile", choices=tuple(PROFILE_MODES),
                        default="qwen3_8b")
    parser.add_argument("--td-proxy-profile", choices=TD_PROXY_PROFILES)
    parser.add_argument(
        "--vram-gate-percent", type=float, default=90.0,
        help="maximum qualified VRAM percentage (default: 90)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not 0.0 < args.vram_gate_percent <= 100.0:
        parser.error("--vram-gate-percent must be in (0, 100]")
    vram_gate_fraction = args.vram_gate_percent / 100.0
    if args.td_proxy_profile and args.benchmark_profile != "qwen3_32b":
        raise ValueError(
            "--td-proxy-profile requires --benchmark-profile qwen3_32b")
    root = Path(args.result_dir)
    modes = PROFILE_MODES[args.benchmark_profile]
    payloads = load_payloads(root, modes)
    for (mode, dataset), payload in payloads.items():
        validate_formal_payload(
            payload, mode, dataset,
            benchmark_profile=args.benchmark_profile,
            qualification_track="canonical", target_alignment="strict",
            td_proxy_profile=None)
    canonical_target_provenance = None
    canonical_fingerprint = None
    if args.benchmark_profile in STRICT_PROFILE_SOURCE_IDENTITIES:
        fingerprints = {target_fingerprint(payload)
                        for payload in payloads.values()}
        if len(fingerprints) != 1:
            raise ValueError(
                f"canonical {args.benchmark_profile} payloads use different "
                "target artifacts")
        canonical_fingerprint = fingerprints.pop()
        canonical_target_provenance = next(iter(
            payloads.values()))["contract"]["target_provenance"]
    report = {
        "phase": args.phase,
        "benchmark_profile": args.benchmark_profile,
        "canonical_modes": list(modes),
        "datasets": {},
        "all_exact_parity": True,
        "vram_gate_percent": args.vram_gate_percent,
    }
    if canonical_target_provenance is not None:
        report["target_provenance"] = canonical_target_provenance
    for dataset in DATASETS:
        report["datasets"][dataset] = {}
        for mode in modes[1:]:
            current_payload = payloads[(mode, dataset)]
            row = evaluate_mode(
                payloads[("ar", dataset)], current_payload, mode, dataset,
                args.phase, literature_gate=args.benchmark_profile == "qwen3_8b",
                vram_gate_fraction=vram_gate_fraction)
            report["all_exact_parity"] &= row["exact_parity"]
            report["datasets"][dataset][mode] = row
    report["hard_gate"] = report["all_exact_parity"] and all(
        report["datasets"][dataset][mode]["hard_gate"]
        for dataset in DATASETS for mode in modes[1:])
    if args.benchmark_profile == "qwen3_8b":
        report["td_stretch_gate"] = all(
            report["datasets"][dataset]["pard2-td"]["stretch_gate"]
            for dataset in LITERATURE)
    elif args.benchmark_profile == "qwen3_32b":
        # Cross-target TD is deliberately isolated from the canonical AR+TI
        # gate and never compared with Qwen3-8B literature thresholds.
        report["td_stretch_gate"] = None
        report["experimental"] = {
            "included": args.td_proxy_profile is not None,
            "td_proxy_profile": args.td_proxy_profile,
            "datasets": {},
        }
        if args.td_proxy_profile is not None:
            proxy_payloads = load_payloads(root, ("pard2-td",))
            for (mode, dataset), payload in proxy_payloads.items():
                validate_formal_payload(
                    payload, mode, dataset,
                    benchmark_profile="qwen3_32b",
                    qualification_track="experimental",
                    target_alignment="cross_target_proxy",
                    td_proxy_profile=args.td_proxy_profile)
                if target_fingerprint(payload) != canonical_fingerprint:
                    raise ValueError(
                        "experimental TD payload uses a different "
                        "Qwen3-32B target artifact")
                row = evaluate_mode(
                    payloads[("ar", dataset)], payload, mode, dataset,
                    args.phase, literature_gate=False,
                    vram_gate_fraction=vram_gate_fraction)
                report["experimental"]["datasets"][dataset] = {
                    "pard2-td": row}
            report["experimental"]["all_exact_parity"] = all(
                report["experimental"]["datasets"][dataset]["pard2-td"][
                    "exact_parity"]
                for dataset in DATASETS)
            report["experimental"]["hard_gate"] = all(
                report["experimental"]["datasets"][dataset]["pard2-td"][
                    "hard_gate"]
                for dataset in DATASETS)
    else:
        # AMD does not publish a Qwen3-14B literature speedup threshold.
        report["td_stretch_gate"] = None
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
