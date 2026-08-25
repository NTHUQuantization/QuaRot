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
LITERATURE = {
    # Official AMD vLLM-v1 Qwen3-8B table. It does not report Qwen MATH-500
    # or acceptance length, so those literature fractions are not invented.
    "humaneval": {"speedup": 6.75},
    "gsm8k": {"speedup": 6.44},
}


def keyed(payload):
    return {(row["sweep"], row["prompt_index"]): row for row in payload["runs"]}


def bootstrap_ci(values, samples=10_000, seed=0x50415244):
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


def validate_formal_payload(payload, mode, dataset):
    contract = payload.get("contract", {})
    expected = {
        "mode": mode, "dataset": dataset, "qualified": True,
        "prompt_count": PROMPT_COUNTS[dataset], "generated_tokens": 256,
        "batch_size": 1, "warmups": 8, "sweeps": 3, "greedy": True,
    }
    mismatches = {
        key: (contract.get(key), value) for key, value in expected.items()
        if contract.get(key) != value}
    expected_runs = PROMPT_COUNTS[dataset] * 3
    if len(payload.get("runs", ())) != expected_runs:
        mismatches["runs"] = (len(payload.get("runs", ())), expected_runs)
    if mismatches:
        raise ValueError(
            f"{mode}/{dataset} is not a formal benchmark payload: {mismatches}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--phase", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.result_dir)
    payloads = {(mode, dataset): json.loads(
        (root / f"{mode}_{dataset}.json").read_text())
        for mode in MODES for dataset in DATASETS}
    for (mode, dataset), payload in payloads.items():
        validate_formal_payload(payload, mode, dataset)
    report = {"phase": args.phase, "datasets": {}, "all_exact_parity": True}
    for dataset in DATASETS:
        ar = keyed(payloads[("ar", dataset)])
        report["datasets"][dataset] = {}
        for mode in ("pard2-ti", "pard2-td"):
            current_payload = payloads[(mode, dataset)]
            current = keyed(current_payload)
            same_keys = ar.keys() == current.keys()
            parity = same_keys and all(ar[key]["output_ids"] == current[key]["output_ids"]
                                       for key in ar)
            report["all_exact_parity"] &= parity
            speedups = [current[key]["steady_tokens_per_s"] / ar[key]["steady_tokens_per_s"]
                        for key in ar if ar[key]["steady_tokens_per_s"] > 0]
            ci = bootstrap_ci(speedups)
            cv = cv_by_sweep(current_payload["runs"], "steady_tokens_per_s")
            peak = max(row["peak_vram_bytes"] for row in current_payload["runs"])
            total = current_payload["gpu_preflight"]["total_bytes"]
            mean_accept = statistics.mean(row["mean_accept_length"]
                                           for row in current_payload["runs"])
            row = {"exact_parity": parity, "median_speedup": statistics.median(speedups),
                   "speedup_bootstrap_95_ci": ci, "run_level_cv": cv,
                   "peak_vram_bytes": peak, "vram_headroom": 1.0 - peak / total,
                   "mean_accept_length": mean_accept,
                   "hard_gate": parity and ci[0] > 1.0 and cv < 0.05 and peak <= 0.90 * total}
            if mode == "pard2-td":
                reference = LITERATURE.get(dataset)
                if reference is not None:
                    row["literature_speed_fraction"] = row["median_speedup"] / reference["speedup"]
                    threshold = 0.70 if args.phase == 1 else 0.80
                    row["stretch_gate"] = row["literature_speed_fraction"] >= threshold
                else:
                    row["stretch_gate"] = None
            report["datasets"][dataset][mode] = row
    report["hard_gate"] = report["all_exact_parity"] and all(
        report["datasets"][dataset][mode]["hard_gate"]
        for dataset in DATASETS for mode in ("pard2-ti", "pard2-td"))
    report["td_stretch_gate"] = all(
        report["datasets"][dataset]["pard2-td"]["stretch_gate"]
        for dataset in LITERATURE)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
