#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


VARIANTS = ["unfused_all", "k1_fused", "k1_k2_fused", "attention_fused", "full_fused"]
ITERS = 20


def read_stats(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def short_name(name):
    markers = [
        "append_kv_had_quant_kernel",
        "BatchDecodeWithPagedKVCacheKernel",
        "output_had_quant_kernel",
        "fused_ffn_silu_hadamard_quant_kernel",
        "reduce_kernel",
        "vectorized_elementwise_kernel",
        "elementwise_kernel_manual_unroll",
    ]
    for marker in markers:
        if marker in name:
            return marker
    if name.startswith("__amd_rocclr"):
        return name
    return name[:64]


def summarize(root):
    rows = []
    details = {}
    for variant in VARIANTS:
        stats = read_stats(root / f"{variant}_kernel_stats.csv")
        total_ns = sum(float(r["TotalDurationNs"]) for r in stats)
        calls = sum(int(r["Calls"]) for r in stats)
        rows.append((variant, total_ns / ITERS / 1000.0, calls / ITERS))
        top = sorted(stats, key=lambda r: float(r["TotalDurationNs"]), reverse=True)[:6]
        details[variant] = [
            (short_name(r["Name"]), float(r["TotalDurationNs"]) / ITERS / 1000.0, int(r["Calls"]) / ITERS)
            for r in top
        ]
    return rows, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="rocprof_model_integration_ablation")
    args = parser.parse_args()
    rows, details = summarize(Path(args.root))
    baseline = rows[0][1]

    print("| Variant | rocprof kernel time / iter (us) | Kernel calls / iter | Speedup vs unfused |")
    print("| --- | ---: | ---: | ---: |")
    for variant, time_us, calls in rows:
        print(f"| {variant} | {time_us:.3f} | {calls:.2f} | {baseline / time_us:.2f}x |")

    print()
    for variant in VARIANTS:
        print(f"### {variant}")
        print("| Top kernel group | Time / iter (us) | Calls / iter |")
        print("| --- | ---: | ---: |")
        for name, time_us, calls in details[variant]:
            print(f"| {name} | {time_us:.3f} | {calls:.2f} |")
        print()


if __name__ == "__main__":
    main()
