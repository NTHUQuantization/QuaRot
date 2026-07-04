#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def short_name(name: str) -> str:
    name = name.strip('"')
    replacements = [
        ("void (anonymous namespace)::", ""),
        ("at::native::", ""),
        ("std::", ""),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    return name[:96]


def read_kernel_stats(path: Path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel = row.get("KernelName") or row.get("Name") or row.get("Kernel_Name")
            calls = row.get("Calls") or row.get("TotalCalls") or row.get("Count")
            total_ns = row.get("TotalDurationNs") or row.get("TotalDuration") or row.get("DurationNs")
            avg_ns = row.get("AverageNs") or row.get("AverageDurationNs") or row.get("AvgDurationNs")
            if kernel is None or calls is None or total_ns is None:
                values = list(row.values())
                if len(values) >= 4:
                    kernel, calls, total_ns = values[0], values[1], values[2]
                    avg_ns = values[3] if avg_ns is None else avg_ns
            try:
                calls_i = int(float(calls))
                total_us = float(total_ns) / 1000.0
                avg_us = float(avg_ns) / 1000.0 if avg_ns is not None else total_us / max(calls_i, 1)
            except (TypeError, ValueError):
                continue
            rows.append({
                "kernel": short_name(kernel),
                "calls": calls_i,
                "total_us": total_us,
                "avg_us": avg_us,
            })
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return rows


def parse_variant(path: Path):
    stem = path.name.replace("_kernel_stats.csv", "")
    match = re.match(r"(?P<variant>.+)_B(?P<batch>\d+)_L(?P<seq>\d+)_H(?P<ffn>\d+)", stem)
    if not match:
        return stem, "", "", ""
    return match.group("variant"), match.group("batch"), match.group("seq"), match.group("ffn")


def write_table(f, rows, columns):
    f.write("| " + " | ".join(columns) + " |\n")
    f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.3f}"
            vals.append(str(val))
        f.write("| " + " | ".join(vals) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="rocprof_single_decoder_layer")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.dir)
    summaries = []
    details = {}
    for stats in sorted(root.glob("*_kernel_stats.csv")):
        variant, batch, seq, ffn = parse_variant(stats)
        rows = read_kernel_stats(stats)
        if not rows:
            continue
        total_us = sum(r["total_us"] for r in rows)
        calls = sum(r["calls"] for r in rows)
        summaries.append({
            "variant": variant,
            "batch": batch,
            "seq_len": seq,
            "ffn_hidden": ffn,
            "kernel_calls_per_iter": calls / args.iters,
            "kernel_time_us_per_iter": total_us / args.iters,
        })
        details[variant] = rows[: args.top]

    out = Path(args.out) if args.out else root / "summary.md"
    with out.open("w") as f:
        f.write("# Single Decoder Layer rocprof Summary\n\n")
        f.write(f"`iters={args.iters}`; times below divide rocprof kernel totals by iterations.\n\n")
        write_table(
            f,
            summaries,
            ["variant", "batch", "seq_len", "ffn_hidden", "kernel_calls_per_iter", "kernel_time_us_per_iter"],
        )
        for variant, rows in details.items():
            f.write(f"\n## Top kernels: {variant}\n\n")
            write_table(f, rows, ["kernel", "calls", "total_us", "avg_us"])


if __name__ == "__main__":
    main()
