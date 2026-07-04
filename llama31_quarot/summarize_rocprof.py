#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def short_name(name):
    name = name.strip('"')
    for prefix in ["void ", "at::native::", "std::"]:
        name = name.replace(prefix, "")
    return name[:100]


def read_stats(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = list(row.values())
            kernel = row.get("KernelName") or row.get("Name") or vals[0]
            calls = row.get("Calls") or vals[1]
            total_ns = row.get("TotalDurationNs") or vals[2]
            avg_ns = row.get("AverageNs") or vals[3]
            try:
                calls = int(float(calls))
                total_us = float(total_ns) / 1000.0
                avg_us = float(avg_ns) / 1000.0
            except (ValueError, TypeError):
                continue
            rows.append({
                "kernel": short_name(kernel),
                "calls": calls,
                "total_us": total_us,
                "avg_us": avg_us,
            })
    return sorted(rows, key=lambda r: r["total_us"], reverse=True)


def table(f, rows, cols):
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for row in rows:
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.3f}"
            vals.append(str(val))
        f.write("| " + " | ".join(vals) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="llama31_full_model_rocprof")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.dir)
    out = Path(args.out) if args.out else root / "summary.md"
    summary = []
    details = {}
    for path in sorted(root.glob("*_kernel_stats.csv")):
        rows = read_stats(path)
        if not rows:
            continue
        name = path.name.replace("_kernel_stats.csv", "")
        total_us = sum(r["total_us"] for r in rows)
        calls = sum(r["calls"] for r in rows)
        summary.append({
            "workload": name,
            "kernel_calls_per_iter": calls / args.iters,
            "kernel_time_us_per_iter": total_us / args.iters,
        })
        details[name] = rows[: args.top]
    with out.open("w") as f:
        f.write("# Llama-3.1 Full Model rocprof Summary\n\n")
        f.write(f"`iters={args.iters}`; totals divide rocprof kernel stats by iterations.\n\n")
        table(f, summary, ["workload", "kernel_calls_per_iter", "kernel_time_us_per_iter"])
        for name, rows in details.items():
            f.write(f"\n## Top kernels: {name}\n\n")
            table(f, rows, ["kernel", "calls", "total_us", "avg_us"])


if __name__ == "__main__":
    main()
