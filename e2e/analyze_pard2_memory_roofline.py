#!/usr/bin/env python3
"""Summarize formal PARD2 VRAM and generate memory/roofline SVGs.

The roofline points are analytical upper bounds.  They intentionally do not
pretend that rocprof counter values are available: gfx1201/ROCm 7.2 returned
zero FETCH_SIZE for the recorded 8 MiB and 24 MiB W4A4 probes.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "e2e" / "assets"
OUT = ROOT / "pard2_optimization_results" / "memory_roofline_summary.json"
PROFILE = ROOT / "pard2_optimization_results" / "roofline_profile_probe"
MEASURED = ROOT / "pard2_optimization_results" / "measured_roofline_summary.json"
GIB = 1024**3

DATASETS = {
    "HumanEval": "humaneval",
    "GSM8K": "gsm8k",
    "MATH-500": "math_500",
}
SERIES = {
    "AR": ("pard2_formal_results_hip_norm", "ar"),
    "TI unoptimized": ("pard2_formal_results_hip_norm", "pard2-ti"),
    "TD unoptimized": ("pard2_formal_results_hip_norm", "pard2-td"),
    "TI mainline": ("pard2_optimization_results/formal_mainline", "pard2-ti"),
    "TD mainline": ("pard2_optimization_results/formal_mainline", "pard2-td"),
}


def formal_path(folder: str, mode: str, dataset: str) -> Path:
    return ROOT / folder / f"{mode}_{dataset}.json"


def memory_summary() -> dict:
    result = {}
    for dataset_name, dataset_slug in DATASETS.items():
        rows = {}
        for label, (folder, mode) in SERIES.items():
            path = formal_path(folder, mode, dataset_slug)
            payload = json.loads(path.read_text())
            peaks = [int(run["peak_vram_bytes"]) for run in payload["runs"]]
            total = int(payload["gpu_preflight"]["total_bytes"])
            maximum = max(peaks)
            rows[label] = {
                "source": str(path.relative_to(ROOT)),
                "runs": len(peaks),
                "median_peak_gib": statistics.median(peaks) / GIB,
                "maximum_peak_gib": maximum / GIB,
                "minimum_headroom_percent": 100 * (1 - maximum / total),
            }
        result[dataset_name] = rows
    return result


def trace_summary(prefix: str, warmups: int, m1_count: int, m16_count: int) -> dict:
    trace = PROFILE / f"{prefix}_kernel_trace.csv"
    counters = PROFILE / f"{prefix}_counter_collection.csv"
    if not trace.exists() or not counters.exists():
        return {"available": False}
    durations = []
    with trace.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "bpre_kernel" in row["Kernel_Name"]:
                durations.append(
                    (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000
                )
    with counters.open(newline="") as handle:
        fetch = [float(row["Counter_Value"]) for row in csv.DictReader(handle)]
    start = warmups
    m1 = durations[start : start + m1_count]
    m16 = durations[start + m1_count : start + m1_count + m16_count]
    return {
        "available": True,
        "kernel": "gfx12 INT4 bpre_kernel<1,4,1,2>",
        "m1_median_us": statistics.median(m1),
        "m16_median_us": statistics.median(m16),
        "m16_over_m1": statistics.median(m16) / statistics.median(m1),
        "fetch_size_counter_values_kib": sorted(set(fetch)),
        "fetch_size_usable": any(value > 0 for value in fetch),
        "note": "FETCH_SIZE=0 is inconsistent with operand size; do not use as DRAM traffic.",
    }


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


def memory_svg(memory: dict) -> str:
    width, height = 1120, 620
    left, top, plot_w, plot_h = 90, 70, 970, 420
    colors = ["#667085", "#84ADFF", "#175CD3", "#6CE9A6", "#039855"]
    labels = list(SERIES)
    ymax = 10.0
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#101828}.title{font-size:24px;font-weight:700}.axis{font-size:14px}.small{font-size:12px}.grid{stroke:#E4E7EC;stroke-width:1}</style>',
        '<text class="title" x="90" y="38">Formal peak VRAM (maximum across all runs)</text>',
    ]
    for tick in range(0, 11, 2):
        y = top + plot_h * (1 - tick / ymax)
        out.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        out.append(f'<text class="axis" x="{left - 16}" y="{y + 5:.1f}" text-anchor="end">{tick}</text>')
    out.append(f'<text class="axis" transform="translate(24 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Peak VRAM (GiB)</text>')
    group_w = plot_w / len(DATASETS)
    bar_w = 43
    gap = 8
    for di, dataset in enumerate(DATASETS):
        base_x = left + di * group_w + 28
        for si, label in enumerate(labels):
            value = memory[dataset][label]["maximum_peak_gib"]
            h = value / ymax * plot_h
            x = base_x + si * (bar_w + gap)
            y = top + plot_h - h
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{h:.1f}" rx="3" fill="{colors[si]}"/>')
            out.append(f'<text class="small" x="{x + bar_w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle">{value:.2f}</text>')
        out.append(f'<text class="axis" x="{left + (di + 0.5) * group_w:.1f}" y="{top + plot_h + 30}" text-anchor="middle">{esc(dataset)}</text>')
    legend_y = 555
    for si, label in enumerate(labels):
        x = 90 + si * 195
        out.append(f'<rect x="{x}" y="{legend_y - 13}" width="16" height="16" rx="2" fill="{colors[si]}"/>')
        out.append(f'<text class="axis" x="{x + 23}" y="{legend_y}">{esc(label)}</text>')
    out.append('<text class="small" x="90" y="602" fill="#475467">AR is the original fused_v1 baseline; headroom remains at least 72.7% for all PARD2 formal runs.</text>')
    out.append('</svg>')
    return "\n".join(out)


def roofline_svg(measured: dict) -> str:
    width, height = 1120, 690
    left, top, plot_w, plot_h = 105, 70, 930, 500
    xmin, xmax = 0.5, 2000.0
    ymin, ymax = 0.1, 1000.0

    def sx(x: float) -> float:
        return left + (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * plot_w

    def sy(y: float) -> float:
        return top + (math.log10(ymax) - math.log10(y)) / (math.log10(ymax) - math.log10(ymin)) * plot_h

    def polyline(ceiling: float) -> str:
        ridge = ceiling / 0.64
        points = [(xmin, 0.64 * xmin), (ridge, ceiling), (xmax, ceiling)]
        return " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#101828}.title{font-size:24px;font-weight:700}.axis{font-size:14px}.small{font-size:12px}.grid{stroke:#E4E7EC;stroke-width:1}.dash{stroke-dasharray:6 5}</style>',
        '<text class="title" x="105" y="37">R9700 roofs and measured cold-cache PARD2 points</text>',
    ]
    for x in [0.5, 1, 4, 16, 64, 256, 1024, 2000]:
        px = sx(x)
        out.append(f'<line class="grid" x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}"/>')
        out.append(f'<text class="axis" x="{px:.1f}" y="{top + plot_h + 25}" text-anchor="middle">{x:g}</text>')
    for y in [0.1, 1, 10, 100, 1000]:
        py = sy(y)
        out.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}"/>')
        out.append(f'<text class="axis" x="{left - 14}" y="{py + 5:.1f}" text-anchor="end">{y:g}</text>')
    out.append(f'<polyline points="{polyline(766)}" fill="none" stroke="#D92D20" stroke-width="3"/>')
    out.append(f'<polyline points="{polyline(191)}" fill="none" stroke="#7F56D9" stroke-width="3"/>')
    out.append(f'<text class="axis" x="{sx(1350):.1f}" y="{sy(766) - 10:.1f}" fill="#D92D20">INT4 WMMA: 766 TOPS</text>')
    out.append(f'<text class="axis" x="{sx(450):.1f}" y="{sy(191) - 10:.1f}" fill="#7F56D9">BF16/FP16 matrix: 191 TFLOP/s</text>')
    out.append(f'<text class="axis" x="{sx(2):.1f}" y="{sy(1.28) - 12:.1f}" transform="rotate(-28 {sx(2):.1f} {sy(1.28) - 12:.1f})" fill="#475467">640 GB/s memory roof</text>')

    analytical = [
        (1.0, 0.64, "RMSNorm/quant region", "#F79009"),
        (3.8739, 0.64 * 3.8739, "M1 640-GB/s roof", "#175CD3"),
        (14.7701, 0.64 * 14.7701, "BF16 640-GB/s roof", "#7F56D9"),
        (61.9823, 0.64 * 61.9823, "M16 640-GB/s roof", "#039855"),
    ]
    for x, y, label, color in analytical:
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="7" fill="#fff" stroke="{color}" stroke-width="2"/>')
        out.append(f'<text class="small" x="{sx(x) + 10:.1f}" y="{sy(y) - 12:.1f}" fill="{color}">{esc(label)}</text>')

    w4 = measured["w4a4_4096x12288"]
    bf16 = measured["bf16_projection_15x16384x1024"]["point"]
    measured_points = [
        (w4["m1"]["operational_intensity_op_per_byte"], w4["m1"]["measured_throughput_top_s"], "M1 measured", "#175CD3"),
        (bf16["operational_intensity_op_per_byte"], bf16["measured_throughput_top_s"], "BF16 measured", "#7F56D9"),
        (w4["m16"]["operational_intensity_op_per_byte"], w4["m16"]["measured_throughput_top_s"], "M16 measured", "#039855"),
    ]
    for x, y, label, color in measured_points:
        roof_y = 0.64 * x
        out.append(f'<line class="dash" x1="{sx(x):.1f}" y1="{sy(y):.1f}" x2="{sx(x):.1f}" y2="{sy(roof_y):.1f}" stroke="{color}" stroke-width="1.5"/>')
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="7" fill="{color}" stroke="#fff" stroke-width="2"/>')
        out.append(f'<text class="axis" x="{sx(x) + 10:.1f}" y="{sy(y) + 18:.1f}" fill="{color}">{esc(label)} {y:.2f} TOP/s</text>')
    out.append(f'<line class="dash" x1="{sx(measured_points[0][0]):.1f}" y1="{sy(measured_points[0][1]):.1f}" x2="{sx(measured_points[2][0]):.1f}" y2="{sy(measured_points[2][1]):.1f}" stroke="#175CD3" stroke-width="2" marker-end="url(#arrow)"/>')
    out.insert(2, '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#175CD3"/></marker></defs>')
    out.append(f'<text class="axis" x="{left + plot_w / 2}" y="{height - 55}" text-anchor="middle">Operational intensity (useful op / algorithmic byte, log scale)</text>')
    out.append(f'<text class="axis" transform="translate(25 {top + plot_h / 2}) rotate(-90)" text-anchor="middle">Measured useful throughput (TOP/s, log scale)</text>')
    out.append('<text class="small" x="105" y="660" fill="#475467">Filled = measured HIP-event latency + exact tensor bytes; hollow = 640-GB/s roof. GL2C DRAM PMCs returned zero and are not used.</text>')
    out.append('</svg>')
    return "\n".join(out)


def main() -> None:
    memory = memory_summary()
    measured = json.loads(MEASURED.read_text())
    profile = {
        "4096x4096": trace_summary("m16_probe", 5, 20, 20),
        "4096x12288_qwen3_ffn": trace_summary("qwen_ffn_probe", 3, 10, 10),
    }
    summary = {
        "formal_memory": memory,
        "hardware": {
            "model": "AMD Radeon AI PRO R9700 / gfx1201",
            "memory_gb": 32,
            "peak_bandwidth_gb_s": 640,
            "fp16_matrix_tflop_s": 191,
            "int4_matrix_top_s": 766,
            "fp16_ridge_flop_per_byte": 191000 / 640,
            "int4_ridge_op_per_byte": 766000 / 640,
            "source": "https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html",
        },
        "analytical_operational_intensity_upper_bounds": {
            "w4a4_ar_useful_m1": 4,
            "w4a4_m16": 64,
            "bf16_drafter_m15": 15,
            "note": "Weights dominate; activation/output/scales lower real OI.",
        },
        "microbenchmark": profile,
        "measured_roofline": measured,
    }
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    (ASSETS / "pard2_formal_peak_vram.svg").write_text(memory_svg(memory) + "\n")
    (ASSETS / "pard2_measured_roofline.svg").write_text(roofline_svg(measured) + "\n")
    print(OUT.relative_to(ROOT))
    print((ASSETS / "pard2_formal_peak_vram.svg").relative_to(ROOT))
    print((ASSETS / "pard2_measured_roofline.svg").relative_to(ROOT))


if __name__ == "__main__":
    main()
