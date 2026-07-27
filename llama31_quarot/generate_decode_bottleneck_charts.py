#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = ["unfused_INT4", "fused_current", "fused_hadacore256"]
VARIANT_LABELS = {
    "unfused_INT4": "Unfused INT4",
    "fused_current": "Fused current",
    "fused_hadacore256": "Fused hadacore256",
}
COLORS = {
    "unfused_INT4": "#C44E52",
    "fused_current": "#2A6FBB",
    "fused_hadacore256": "#2A9D6F",
}
LINE_STYLES = {
    "unfused_INT4": "-",
    "fused_current": "-",
    "fused_hadacore256": "--",
}
MARKERS = {
    "unfused_INT4": "s",
    "fused_current": "o",
    "fused_hadacore256": "x",
}
MARKER_SIZES = {
    "unfused_INT4": 6,
    "fused_current": 8,
    "fused_hadacore256": 5,
}
ZORDERS = {
    "unfused_INT4": 1,
    "fused_current": 2,
    "fused_hadacore256": 3,
}
CONTEXTS = [10, 128, 1024, 4096]
BATCHES = [1, 2, 4, 8]
PEAK_FP16_MATRIX_TFLOPS = 191.0
PEAK_MEMORY_GBPS = 640.0


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fvalue(row, key):
    return float(row[key])


def ivalue(row, key):
    return int(float(row[key]))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax):
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_variant(ax, x, y, variant):
    kwargs = {
        "marker": MARKERS[variant],
        "markersize": MARKER_SIZES[variant],
        "linestyle": LINE_STYLES[variant],
        "linewidth": 2,
        "color": COLORS[variant],
        "label": VARIANT_LABELS[variant],
        "zorder": ZORDERS[variant],
    }
    if variant == "fused_current":
        kwargs.update(markerfacecolor="white", markeredgewidth=2)
    elif variant == "fused_hadacore256":
        kwargs.update(markeredgewidth=2)
    ax.plot(x, y, **kwargs)


def save(fig, path, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def latency_context_chart(rows, chart_dir, variants):
    lookup = {(r["variant"], ivalue(r, "batch"), ivalue(r, "context_len")): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True)
    for ax, batch in zip(axes.flat, BATCHES):
        for variant in variants:
            values = [fvalue(lookup[(variant, batch, context)], "mean") for context in CONTEXTS]
            plot_variant(ax, CONTEXTS, values, variant)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONTEXTS, [str(v) for v in CONTEXTS])
        ax.set_title(f"Batch = {batch}")
        ax.set_ylabel("Decode latency (ms/token)")
        style_axis(ax)
    axes[1, 0].set_xlabel("Context length")
    axes[1, 1].set_xlabel("Context length")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.suptitle("Decode latency vs. context length", y=0.995, fontsize=14)
    save(fig, chart_dir / "latency_vs_context_by_batch.png", rect=(0, 0, 1, 0.88))


def speedup_context_chart(rows, chart_dir, variants):
    lookup = {(r["variant"], ivalue(r, "batch"), ivalue(r, "context_len")): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True)
    for ax, batch in zip(axes.flat, BATCHES):
        for variant in [v for v in variants if v != "unfused_INT4"]:
            values = [fvalue(lookup[(variant, batch, context)], "speedup_vs_unfused_INT4") for context in CONTEXTS]
            plot_variant(ax, CONTEXTS, values, variant)
        ax.axhline(1.0, color="#777777", linewidth=1, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(CONTEXTS, [str(v) for v in CONTEXTS])
        ax.set_title(f"Batch = {batch}")
        ax.set_ylabel("Speedup vs. unfused INT4")
        style_axis(ax)
    axes[1, 0].set_xlabel("Context length")
    axes[1, 1].set_xlabel("Context length")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.suptitle("Fusion speedup vs. context length", y=0.995, fontsize=14)
    save(fig, chart_dir / "speedup_vs_context_by_batch.png", rect=(0, 0, 1, 0.88))


def latency_batch_chart(rows, chart_dir, variants):
    lookup = {(r["variant"], ivalue(r, "batch"), ivalue(r, "context_len")): r for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True)
    for ax, context in zip(axes.flat, CONTEXTS):
        for variant in variants:
            values = [fvalue(lookup[(variant, batch, context)], "mean") for batch in BATCHES]
            plot_variant(ax, BATCHES, values, variant)
        ax.set_xticks(BATCHES)
        ax.set_title(f"Context = {context}")
        ax.set_ylabel("Decode latency (ms/token)")
        style_axis(ax)
    axes[1, 0].set_xlabel("Batch")
    axes[1, 1].set_xlabel("Batch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False)
    fig.suptitle("Decode latency vs. batch", y=0.995, fontsize=14)
    save(fig, chart_dir / "latency_vs_batch_by_context.png", rect=(0, 0, 1, 0.88))


def stacked_chart(rows, key_field, value_field, workloads, categories, labels, colors, title, output):
    lookup = defaultdict(float)
    for row in rows:
        lookup[(row[key_field], row["category"])] += fvalue(row, value_field)
    fig, ax = plt.subplots(figsize=(13, 6.2))
    bottom = [0.0] * len(workloads)
    for category in categories:
        values = [lookup[(workload, category)] for workload in workloads]
        ax.bar(range(len(workloads)), values, bottom=bottom, label=labels.get(category, category), color=colors[category], width=0.72)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(range(len(workloads)), [w.replace("_B", "\nB").replace("_L", ", L") for w in workloads], rotation=0)
    ax.set_ylabel("Share of summed GPU kernel time (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    style_axis(ax)
    save(fig, output)


def rocprof_category_chart(rows, chart_dir, variants):
    workloads = []
    for shape in ["B1_L128", "B1_L4096", "B4_L1024"]:
        workloads.extend([f"{variant}_{shape}" for variant in variants])
    categories = ["rocBLAS_GEMM", "K2_INT4", "K2_FP16", "PyTorch_elementwise_copy_reduce", "fused_other", "other"]
    normalized = []
    for row in rows:
        category = row["category"]
        if category in {"K1", "K3", "FFN"}:
            category = "fused_other"
        elif category not in categories:
            category = "other"
        normalized.append({**row, "category": category})
    stacked_chart(
        normalized,
        "workload",
        "percent_of_kernel_sum",
        workloads,
        categories,
        {
            "rocBLAS_GEMM": "rocBLAS GEMM",
            "K2_INT4": "K2 direct INT4",
            "K2_FP16": "K2 FP16",
            "PyTorch_elementwise_copy_reduce": "PyTorch elementwise/copy/reduce",
            "fused_other": "K1/K3/FFN fused",
            "other": "Other",
        },
        {
            "rocBLAS_GEMM": "#2A6FBB",
            "K2_INT4": "#E9A23B",
            "K2_FP16": "#F2C66D",
            "PyTorch_elementwise_copy_reduce": "#C44E52",
            "fused_other": "#2A9D6F",
            "other": "#8C8C8C",
        },
        "rocprof decode-only kernel-time composition",
        chart_dir / "rocprof_kernel_category_stacked.png",
    )


def projection_weight_elements():
    hidden = 4096
    intermediate = 14336
    kv_hidden = 1024
    vocab = 128256
    per_layer = (
        hidden * hidden * 2
        + hidden * kv_hidden * 2
        + hidden * intermediate * 3
    )
    return per_layer * 32 + hidden * vocab


def gemm_bound_rows(rocprof_rows):
    weights = projection_weight_elements()
    weight_bytes = weights * 2
    rows = []
    for row in rocprof_rows:
        if row["category"] != "rocBLAS_GEMM" or not row["workload"].startswith("fused_current_"):
            continue
        batch = int(row["workload"].split("_B", 1)[1].split("_", 1)[0])
        context = int(row["workload"].rsplit("_L", 1)[1])
        kernel_ms = fvalue(row, "kernel_sum_ms_per_token")
        flops = 2 * weights * batch
        arithmetic_intensity = flops / weight_bytes
        rows.append({
            "variant": "fused_current",
            "batch": batch,
            "context_len": context,
            "rocblas_kernel_ms_per_token": kernel_ms,
            "fp16_weight_bytes_lower_bound": weight_bytes,
            "algorithmic_flops": flops,
            "arithmetic_intensity_flop_per_byte_upper_bound": arithmetic_intensity,
            "effective_tflops": flops / kernel_ms / 1e9,
            "effective_weight_bandwidth_GBps": weight_bytes / kernel_ms / 1e6,
            "percent_of_peak_memory_bandwidth": 100.0 * (weight_bytes / kernel_ms / 1e6) / PEAK_MEMORY_GBPS,
            "hardware_ridge_point_flop_per_byte": PEAK_FP16_MATRIX_TFLOPS * 1000.0 / PEAK_MEMORY_GBPS,
            "bound_classification": "memory/weight-streaming bound at measured batch",
        })
    return sorted(rows, key=lambda r: (r["batch"], r["context_len"]))


def roofline_chart(rows, chart_dir):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    xs = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    roof = [min(PEAK_FP16_MATRIX_TFLOPS, PEAK_MEMORY_GBPS * x / 1000.0) for x in xs]
    ax.plot(xs, roof, color="#555555", linewidth=2, label="R9700 theoretical FP16 roofline")
    point_styles = {
        (1, 128): {"marker": "+", "s": 150, "color": "#2A6FBB", "linewidths": 2.5, "zorder": 3},
        (1, 4096): {"marker": "x", "s": 70, "color": "#E07A24", "linewidths": 2.5, "zorder": 4},
        (4, 1024): {"marker": "D", "s": 85, "color": "#2A9D6F", "zorder": 3},
    }
    annotation_offsets = {
        (1, 128): (12, 12),
        (1, 4096): (12, -22),
        (4, 1024): (12, -4),
    }
    for row in rows:
        key = (row["batch"], row["context_len"])
        label = f"B={row['batch']}, L={row['context_len']}"
        x = row["arithmetic_intensity_flop_per_byte_upper_bound"]
        y = row["effective_tflops"]
        ax.scatter(x, y, label=label, **point_styles[key])
        ax.annotate(
            label,
            (x, y),
            xytext=annotation_offsets[key],
            textcoords="offset points",
            fontsize=9,
            color=point_styles[key]["color"],
        )
    ax.axvline(PEAK_FP16_MATRIX_TFLOPS * 1000.0 / PEAK_MEMORY_GBPS, color="#C44E52", linestyle="--", linewidth=1.5, label="Theoretical ridge point")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Arithmetic intensity upper bound (FLOP/byte)")
    ax.set_ylabel("Effective projection throughput (TFLOP/s)")
    ax.set_title("Projection GEMM roofline-style analysis")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax)
    save(fig, chart_dir / "projection_gemm_roofline.png")


def generate(root, include_hadacore=True, chart_dir_name="charts"):
    root = Path(root)
    variants = VARIANTS if include_hadacore else VARIANTS[:2]
    chart_dir = root / chart_dir_name
    chart_dir.mkdir(parents=True, exist_ok=True)
    latency = read_csv(root / "latency_summary.csv")
    rocprof = read_csv(root / "rocprof_category_summary.csv")
    latency_context_chart(latency, chart_dir, variants)
    speedup_context_chart(latency, chart_dir, variants)
    latency_batch_chart(latency, chart_dir, variants)
    rocprof_category_chart(rocprof, chart_dir, variants)
    bound_rows = gemm_bound_rows(rocprof)
    write_csv(root / "gemm_bound_analysis.csv", bound_rows)
    roofline_chart(bound_rows, chart_dir)
    return bound_rows


def main():
    parser = argparse.ArgumentParser(description="Generate decode bottleneck charts and GEMM bound analysis")
    parser.add_argument("--dir", default="decode_bottleneck_profiling_results")
    parser.add_argument("--exclude-hadacore", action="store_true")
    parser.add_argument("--chart-dir-name", default="charts")
    args = parser.parse_args()
    generate(
        Path(args.dir),
        include_hadacore=not args.exclude_hadacore,
        chart_dir_name=args.chart_dir_name,
    )


if __name__ == "__main__":
    main()
