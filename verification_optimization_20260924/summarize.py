"""Produce tables from measured files only; fail token/acceptance parity loudly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from run_ab import STAGES


def load(path):
    return json.loads(path.read_text())


def run_key(run):
    return run["sweep"], run["prompt_index"], run["input_ids_sha256"]


def paired_runs(actual, expected):
    if expected is None:
        return False, {}
    references = {run_key(run): run for run in expected}
    actual_keys = [run_key(run) for run in actual]
    coverage = (len(actual) == len(expected) == len(references)
                and len(set(actual_keys)) == len(actual_keys)
                and set(actual_keys) == set(references))
    return coverage, references


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--baseline", default="baseline",
                        help="timing/parity reference stage; baseline_repeat exposes final-source drift")
    args = parser.parse_args()
    root = args.root
    stage_order = {name: index for index, name in enumerate(STAGES)}
    directories = sorted(root.iterdir(), key=lambda p: (stage_order.get(p.name, 999), p.name))
    summary = {"reference_stage": args.baseline,
               "target": [], "pard2": [], "missing": [], "parity_failures": []}
    lines = ["# Verification 最佳化實測表", "",
             "此表僅由已存在的量測 JSON 產生；未完成的 stage 不填入預估收益。",
             "Target latency 以 HIP event 的 median 表示；kernel sum 不等於端到端 latency。",
             "PARD-2 小樣本 smoke 保留原 checkpoint、greedy acceptance 與資料集 hash 檢查；不是正式三資料集 qualification。", "",
             f"## Target-only，相同 context={args.context}", "",
             "|Stage|M|GPU kernel launches|CPU launch APIs|Event median ms|Wall median ms|Strict checks|",
             "|---|---:|---:|---:|---:|---:|---|"]
    baseline_target = root / args.baseline / "target/target.json"
    reference = load(baseline_target) if baseline_target.exists() else None
    for directory in directories:
        path = directory / "target/target.json"
        if not path.is_file():
            continue
        data = load(path)
        for case in data["cases"].values():
            if case["context"] != args.context:
                continue
            record = {"stage": directory.name, "M": case["rows"],
                      "launches": case["all_kernel_launches"],
                      "cpu_launch_api_calls": case.get("cpu_launch_api_calls"),
                      "kernel_count_source": case.get("kernel_count_source", "GPU trace"),
                      "event_ms": case["event_ms_median"],
                      "wall_ms": case["wall_ms_median"],
                      "passed": all(case["checks"].values())}
            if reference:
                base = reference["cases"][f"context{args.context}_M{case['rows']}"]
                record["event_ratio_baseline_over_stage"] = base["event_ms_median"] / record["event_ms"]
            summary["target"].append(record)
            lines.append(f"|{directory.name}|{case['rows']}|{record['launches']}|"
                         f"{record['cpu_launch_api_calls']}|"
                         f"{record['event_ms']:.3f}|{record['wall_ms']:.3f}|"
                         f"{'PASS' if record['passed'] else 'FAIL'}|")
            if not record["passed"]:
                summary["parity_failures"].append({"stage": directory.name,
                    "case": f"M{case['rows']}", "checks": case["checks"]})

    lines += ["", "## PARD-2 實際生成", "",
              "|Stage|Mode|Steady tokens/s|E2E tokens/s|Verify ms/step|Mean accept|Output parity|Reject steps|",
              "|---|---|---:|---:|---:|---:|---|---:|"]
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("pard2-*.json")):
            if path.name.endswith(".idle_preflight.json"):
                continue
            data = load(path)
            contract, runs = data["contract"], data["runs"]
            baseline_file = root / args.baseline / path.name
            baseline = load(baseline_file) if baseline_file.exists() else None
            coverage, base_runs = paired_runs(runs, baseline["runs"] if baseline else None)
            token_parity = acceptance_parity = coverage
            for run in runs:
                expected = base_runs.get(run_key(run))
                if expected is None:
                    token_parity = acceptance_parity = False
                    break
                token_parity &= expected["output_ids"] == run["output_ids"]
                acceptance_parity &= expected["accept_length_by_step"] == run["accept_length_by_step"]
            rejected = sum(sum(accepted - 1 < proposed for accepted, proposed in
                               zip(run["accept_length_by_step"], run["proposal_lengths_by_step"]))
                           for run in runs)
            ar_path = root / "baseline" / path.name.replace(contract["mode"], "ar", 1)
            ar_parity = False
            if ar_path.exists():
                ar_coverage, ar_runs = paired_runs(runs, load(ar_path)["runs"])
                ar_parity = ar_coverage and all(run_key(run) in ar_runs and
                                run["output_ids"] == ar_runs[run_key(run)]["output_ids"]
                                for run in runs)
            verify_ms = statistics.median(run["stage_ms"]["target_verify"] / run["verifier_steps"]
                                          for run in runs)
            record = {"stage": directory.name, "mode": contract["mode"],
                      "dataset": contract["dataset"], "qualified": contract["qualified"],
                      "steady_tokens_per_s": data["median_steady_tokens_per_s"],
                      "end_to_end_tokens_per_s": data["median_end_to_end_tokens_per_s"],
                      "verify_ms_per_step": verify_ms,
                      "mean_accept": statistics.mean(run["mean_accept_length"] for run in runs),
                      "output_parity": token_parity, "acceptance_parity": acceptance_parity,
                      "ar_output_parity": ar_parity,
                      "paired_run_coverage": coverage, "run_count": len(runs),
                      "baseline_run_count": len(baseline["runs"]) if baseline else None,
                      "rejected_steps": rejected}
            summary["pard2"].append(record)
            lines.append(f"|{directory.name}|{contract['mode']}|"
                         f"{record['steady_tokens_per_s']:.3f}|{record['end_to_end_tokens_per_s']:.3f}|"
                         f"{verify_ms:.3f}|{record['mean_accept']:.3f}|{token_parity}|{rejected}|")
            if token_parity is False:
                summary["parity_failures"].append({"stage": directory.name,
                                                  "mode": contract["mode"], "output_parity": False})
            if not coverage or not acceptance_parity:
                summary["parity_failures"].append({"stage": directory.name,
                    "mode": contract["mode"], "paired_run_coverage": coverage,
                    "acceptance_parity": acceptance_parity})
            if ar_parity is False:
                summary["parity_failures"].append({"stage": directory.name,
                    "mode": contract["mode"], "ar_output_parity": False})
    lines += ["", "Acceptance 與每輪 accept length 另記錄在 summary.json；輸出 token 相同不代表 candidate 相同。",
              "Eager kernel 數取自 GPU trace；ROCtracer 無法完整列出 graph 內部事件時，以 HIP graph kernel node 枚舉加上 graph 外的 kernel launch API 計數。詳細來源保存在 summary.json；單次 graph API 呼叫不代表單一 GPU kernel。", ""]
    (root / "measurements_zh.md").write_text("\n".join(lines))
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(root / "measurements_zh.md")
    if summary["parity_failures"]:
        raise SystemExit("strict parity failures exist; no adoption claim is valid")


if __name__ == "__main__":
    main()
