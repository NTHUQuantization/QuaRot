#!/usr/bin/env python3
"""Plot three-dataset average PARD2 TI/TD metrics across Qwen3 sizes."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "qwen3_ti_td_comparison"
DATASETS = ("humaneval", "gsm8k", "math_500")
MODES = ("TI", "TD")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def qualification_rows(path: Path, display_size: str) -> list[dict]:
    qualification = load(path)
    rows = []
    for display_mode, json_mode in (("TI", "pard2-ti"), ("TD", "pard2-td")):
        values = [qualification["datasets"][dataset][json_mode] for dataset in DATASETS]
        rows.append(
            {
                "size": display_size,
                "mode": display_mode,
                "mean_accept": np.mean([item["mean_accept_length"] for item in values]),
                "speedup": np.mean([item["median_speedup"] for item in values]),
                "peak_memory_gib": np.mean(
                    [
                        (
                            item["peak_device_used_bytes"] / 2**30
                            if "peak_device_used_bytes" in item
                            else item["peak_vram_bytes"] / 2**30
                        )
                        for item in values
                    ]
                ),
                "evidence": "formal",
                "memory_evidence": (
                    "measured device used"
                    if "peak_device_used_bytes" in values[0]
                    else "legacy allocator peak; replaced by measured smoke"
                ),
            }
        )
    return rows


def rows_32b() -> list[dict]:
    smoke = ROOT / "qwen3_32b_results" / "benchmark"
    formal_qualification = load(
        ROOT / "qwen3_32b_results" / "accuracy" / "formal_8k" / "qualification.json"
    )
    ar_paths = {
        "humaneval": smoke / "smoke" / "ar_humaneval_cache2048_k1off.json",
        "gsm8k": smoke / "smoke" / "ar_gsm8k_cache2048_k1off.json",
        "math_500": smoke / "smoke" / "ar_math_500_cache2048_k1off.json",
    }
    mode_paths = {
        "TD": {
            dataset: smoke
            / "td_proxy_smoke_2k"
            / f"pard2_td_{dataset}_cache2048_tok32_limit1_off0.json"
            for dataset in DATASETS
        },
    }
    ar_tps = {
        dataset: load(path)["runs"][0]["steady_tokens_per_s"]
        for dataset, path in ar_paths.items()
    }
    ti_values = [
        formal_qualification["datasets"][dataset]["pard2-ti"] for dataset in DATASETS
    ]
    rows = [
        {
            "size": "32B",
            "mode": "TI",
            "mean_accept": np.mean([item["mean_accept_length"] for item in ti_values]),
            "speedup": np.mean([item["median_speedup"] for item in ti_values]),
            "peak_memory_gib": np.mean(
                [item["peak_device_used_bytes"] / 2**30 for item in ti_values]
            ),
            "evidence": "formal",
            "memory_evidence": "measured device used",
        }
    ]
    for mode in ("TD",):
        results = {dataset: load(path) for dataset, path in mode_paths[mode].items()}
        runs = {dataset: result["runs"][0] for dataset, result in results.items()}
        rows.append(
            {
                "size": "32B",
                "mode": mode,
                "mean_accept": np.mean(
                    [runs[dataset]["mean_accept_length"] for dataset in DATASETS]
                ),
                "speedup": np.mean(
                    [
                        runs[dataset]["steady_tokens_per_s"] / ar_tps[dataset]
                        for dataset in DATASETS
                    ]
                ),
                "peak_memory_gib": np.mean(
                    [
                        results[dataset]["external_vram_monitor"]["peak_device_used_bytes"]
                        / 2**30
                        for dataset in DATASETS
                    ]
                ),
                "evidence": "smoke (1 prompt/dataset)",
                "memory_evidence": "measured external device used",
            }
        )
    return rows


def plot_metric(rows: list[dict], metric: str, ylabel: str, title: str, filename: str) -> None:
    sizes = ("8B", "14B", "32B")
    x = np.arange(len(sizes))
    width = 0.34
    colors = {"TI": "#3973B8", "TD": "#E68635"}

    fig, ax = plt.subplots(figsize=(9.6, 6.1), constrained_layout=True)
    for index, mode in enumerate(MODES):
        values = [next(row[metric] for row in rows if row["size"] == size and row["mode"] == mode) for size in sizes]
        positions = x + (index - 0.5) * width
        bars = ax.bar(positions, values, width, label=mode, color=colors[mode], edgecolor="#222222", linewidth=0.7)
        for bar, size in zip(bars, sizes):
            evidence = next(
                row["evidence"]
                for row in rows
                if row["size"] == size and row["mode"] == mode
            )
            if evidence.startswith("smoke"):
                bar.set_hatch("///")
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=10)

    if metric == "speedup":
        ax.axhline(1.0, color="#555555", linewidth=1.1, linestyle="--", label="AR baseline")
    ax.set_xticks(x, sizes)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=14, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3 if metric == "speedup" else 2)
    ax.margins(y=0.16)
    fig.text(
        0.5,
        -0.01,
        (
            "Average over HumanEval, GSM8K, and Math-500.  Hatched 32B TD = uncalibrated smoke/proxy; all other bars are formal."
            if metric != "peak_memory_gib"
            else "Total device VRAM used.  8B uses a measured 8K-cache smoke peak; 14B/32B use recorded device peaks."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = qualification_rows(
        ROOT / "pard2_formal_results_hip_norm" / "qualification_phase1.json",
        "8B",
    )
    rows += qualification_rows(
        ROOT / "qwen3_14b_results" / "formal_8k_corrected_20260909" / "qualification.json",
        "14B",
    )
    rows += rows_32b()

    # Legacy 8B formal artifacts predate device-level snapshots. Keep their
    # formal acceptance/speedup, but replace memory with a measured 8K-cache
    # smoke peak from the same target and drafter runtime.
    for mode, filename in (("TI", "pard2_ti.json"), ("TD", "pard2_td.json")):
        measured = load(OUT / "memory_smoke_8b" / filename)
        device_peaks = [
            snapshot["device_used_bytes"]
            for snapshot in measured.get("memory_snapshots", [])
            if snapshot.get("device_used_bytes") is not None
        ]
        device_peaks += [
            run["memory_snapshot"]["device_used_bytes"]
            for run in measured.get("runs", [])
            if (run.get("memory_snapshot") or {}).get("device_used_bytes") is not None
        ]
        external_peak = (measured.get("external_vram_monitor") or {}).get(
            "peak_device_used_bytes"
        )
        if external_peak is not None:
            device_peaks.append(external_peak)
        row = next(row for row in rows if row["size"] == "8B" and row["mode"] == mode)
        row["peak_memory_gib"] = max(device_peaks) / 2**30
        row["memory_evidence"] = "measured device used (8K-cache smoke)"

    with (OUT / "averages.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "averages.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    plot_metric(rows, "mean_accept", "Mean accept length", "PARD2 Mean Accept", "mean_accept.png")
    plot_metric(rows, "speedup", "Paired steady speedup vs AR (x)", "PARD2 Speedup", "speedup.png")
    plot_metric(
        rows,
        "peak_memory_gib",
        "Peak total device VRAM used (GiB)",
        "PARD2 Peak Total VRAM",
        "peak_memory.png",
    )


if __name__ == "__main__":
    main()
