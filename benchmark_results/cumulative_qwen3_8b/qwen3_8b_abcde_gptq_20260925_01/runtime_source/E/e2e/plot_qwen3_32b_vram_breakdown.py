#!/usr/bin/env python3
"""Create a publication-ready VRAM breakdown for Qwen3-32B PARD2-TI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "qwen3_32b_results" / "benchmark" / "gptq_cache_scaling"
OUTPUT = ROOT / "qwen3_32b_results" / "paper_figures" / "qwen3_32b_vram_breakdown.png"
MIB = 2**20
GIB = 2**30


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def stage(payload: dict, name: str) -> dict:
    return next(item for item in payload["memory_snapshots"] if item["stage"] == name)


def external_peak_mib(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(float(row["vram_used"]) for row in csv.DictReader(handle))


def main() -> None:
    ar = load_json(RESULTS / "ar_humaneval_cache8192.json")
    ti = load_json(RESULTS / "pard2_ti_humaneval_cache8192.json")
    external_mib = external_peak_mib(
        RESULTS / "pard2_ti_humaneval_cache8192_amd_smi.csv"
    )
    external = external_mib * MIB

    ar_loaded = stage(ar, "runtime_loaded")["allocated_bytes"]
    ti_loaded = stage(ti, "runtime_loaded")["allocated_bytes"]
    ti_warmup = stage(ti, "warmups_complete")
    peak_active = ti_warmup["max_allocated_bytes"]
    peak_reserved = ti_warmup["max_reserved_bytes"]

    target_cfg = ti["contract"]["target_provenance"]["source_config"]
    cache_len = ti["contract"]["max_cache_len"]
    target_kv = (
        target_cfg["num_hidden_layers"]
        * 2
        * target_cfg["num_key_value_heads"]
        * target_cfg["head_dim"]
        * cache_len
        // 2
    )

    # PARD2-Qwen3-8B: 28 layers, 8 KV heads, head_dim 128, BF16 KV.
    draft_kv = 28 * 2 * 8 * 128 * cache_len * 2
    draft_weights = ti_loaded - ar_loaded
    pard_total = draft_weights + draft_kv
    other_active = peak_active - ar_loaded - target_kv - pard_total
    allocator_reserve = peak_reserved - peak_active
    non_allocator = external - peak_reserved

    total_parts = [
        ("32B target weights", ar_loaded / GIB),
        ("Target KV4 cache", target_kv / GIB),
        ("PARD-2 active", pard_total / GIB),
        ("Other active tensors", other_active / GIB),
        ("Allocator reserve/workspace", allocator_reserve / GIB),
        ("HIP/driver/non-allocator", non_allocator / GIB),
    ]
    pard_parts = [
        ("Drafter weights", draft_weights / GIB),
        ("Drafter BF16 KV cache", draft_kv / GIB),
    ]

    colors = {
        "32B target weights": "#0072B2",
        "Target KV4 cache": "#56B4E9",
        "PARD-2 active": "#D55E00",
        "Other active tensors": "#E69F00",
        "Allocator reserve/workspace": "#999999",
        "HIP/driver/non-allocator": "#4D4D4D",
        "Drafter weights": "#D55E00",
        "Drafter BF16 KV cache": "#F0A17A",
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(7.15, 4.28), constrained_layout=False)
    grid = fig.add_gridspec(
        3,
        1,
        height_ratios=(1.38, 0.42, 1.0),
        left=0.085,
        right=0.985,
        top=0.94,
        bottom=0.17,
        hspace=0.42,
    )
    ax_total = fig.add_subplot(grid[0])
    ax_legend = fig.add_subplot(grid[1])
    ax_pard = fig.add_subplot(grid[2])

    left = 0.0
    handles = []
    for label, value in total_parts:
        ax_total.barh(
            [0],
            [value],
            left=left,
            height=0.48,
            color=colors[label],
            edgecolor="white",
            linewidth=0.45,
        )
        handles.append(
            Patch(
                facecolor=colors[label],
                edgecolor="#333333",
                linewidth=0.45,
                label=f"{label} ({value:.2f} GiB)",
            )
        )
        if label == "32B target weights":
            ax_total.text(
                left + value / 2,
                0,
                f"Target weights\n{value:.2f} GiB (62.4%)",
                ha="center",
                va="center",
                color="white",
                fontsize=8.2,
                fontweight="bold",
            )
        elif label == "PARD-2 active":
            ax_total.text(
                left + value / 2,
                0,
                f"PARD-2\n{value:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=7.2,
                fontweight="bold",
            )
        elif label == "Allocator reserve/workspace":
            ax_total.text(
                left + value / 2,
                0,
                f"Reserve/workspace\n{value:.2f} GiB (25.8%)",
                ha="center",
                va="center",
                color="white",
                fontsize=7.8,
                fontweight="bold",
            )
        left += value

    active_gib = peak_active / GIB
    external_gib = external / GIB
    ax_total.axvline(active_gib, color="#222222", linestyle=(0, (3, 2)), linewidth=0.9)
    ax_total.annotate(
        f"Peak active allocation: {active_gib:.2f} GiB",
        xy=(active_gib, 0.28),
        xytext=(active_gib, 0.62),
        ha="center",
        va="bottom",
        fontsize=7.8,
        arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": "#222222"},
    )
    ax_total.text(
        external_gib,
        -0.39,
        f"Total device peak: {external_gib:.2f} GiB",
        ha="right",
        va="top",
        fontsize=8.0,
        fontweight="bold",
    )
    ax_total.axvline(32, color="#A32638", linewidth=1.2)
    ax_total.text(
        31.65, 0.62, "R9700 limit\n32 GiB (nominal)",
        ha="right", va="bottom", fontsize=7.8, color="#A32638",
    )
    ax_total.set_xlim(0, 33)
    ax_total.set_ylim(-0.55, 0.82)
    ax_total.set_yticks([])
    ax_total.set_xlabel("GPU memory (GiB)")
    ax_total.set_title("(a) Total 8K PARD2-TI device-memory peak", loc="left", fontweight="bold")
    ax_total.set_xticks([0, 8, 16, 24, 32])
    ax_total.grid(axis="x", color="#D0D0D0", linewidth=0.55)
    ax_total.set_axisbelow(True)
    ax_total.spines[["left", "right", "top"]].set_visible(False)
    ax_legend.axis("off")
    ax_legend.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.7,
        handleheight=0.9,
    )

    left = 0.0
    for label, value in pard_parts:
        ax_pard.barh(
            [0],
            [value],
            left=left,
            height=0.48,
            color=colors[label],
            edgecolor="white",
            linewidth=0.5,
        )
        ax_pard.text(
            left + value / 2,
            0,
            f"{label}\n{value:.2f} GiB ({100 * value / (pard_total / GIB):.1f}%)",
            ha="center",
            va="center",
            color="white" if label == "Drafter weights" else "#222222",
            fontsize=8.0,
            fontweight="bold",
        )
        left += value

    ax_pard.set_xlim(0, 2.1)
    ax_pard.set_ylim(-0.54, 0.52)
    ax_pard.set_yticks([])
    ax_pard.set_xlabel("PARD-2 active memory (GiB)")
    ax_pard.set_title(
        "(b) PARD-2 contribution: 1.99 GiB (7.1% of device peak; 9.8% of active allocation)",
        loc="left",
        fontweight="bold",
    )
    ax_pard.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax_pard.grid(axis="x", color="#D0D0D0", linewidth=0.55)
    ax_pard.set_axisbelow(True)
    ax_pard.spines[["left", "right", "top"]].set_visible(False)

    fig.text(
        0.5,
        0.035,
        (
            "Qwen3-32B GPTQ W4A4KV4; batch 1; 8,192-token cache; PARD2-Qwen3-8B drafter.\n"
            "High-water decomposition: component peaks need not coincide. Device-reported capacity: 31.86 GiB."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#333333",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
