#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


ITERS = 20

WORKLOADS = [
    ("FFN", "unfused PyTorch", "ffn_unfused", None, None, None),
    ("FFN", "hadacore pipeline", "ffn_hadacore", None, None, None),
    ("FFN", "fused prototype", "ffn_fused", "fused_ffn_silu_hadamard_quant_kernel", 64624, None),
    ("K1", "unfused PyTorch, RoPE off", "k1_unfused_no_rope", None, None, None),
    ("K1", "unfused PyTorch, RoPE on", "k1_unfused_rope", None, None, None),
    ("K1", "fused append, RoPE off", "k1_no_rope", "append_kv_had_quant_kernel", 5184, None),
    ("K1", "fused append, RoPE on", "k1_rope", "append_kv_had_quant_kernel", 5184, None),
    ("K2", "baseline FP16 KV decode", "k2_f16", "BatchDecodeWithPagedKVCacheKernel", 16777216, None),
    ("K2", "QuaRot original unfused INT4 dequant + FP16 decode", "k2_quarot_unfused", None, None, ITERS),
    ("K2", "optimized INT4 KV decode", "k2_i4", "BatchDecodeWithPagedKVCacheKernel", 4456448, None),
    ("K3", "unfused PyTorch", "k3_unfused", None, None, None),
    ("K3", "fused output quant", "k3", "output_had_quant_kernel", 10272, None),
]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def selected_kernel_rows(path, include, min_calls):
    rows = read_csv(path)
    if include:
        rows = [r for r in rows if include in r["Name"]]
    if min_calls:
        rows = [r for r in rows if int(r["Calls"]) >= min_calls]
    return rows


def selected_counter_rows(path, include):
    rows = read_csv(path)
    if include:
        rows = [r for r in rows if include in r["Kernel_Name"]]
    return rows


def weighted_average(items):
    num = 0.0
    den = 0.0
    for value, weight in items:
        num += value * weight
        den += weight
    return num / den if den else 0.0


def summarize_one(root, slug, include, min_calls):
    stats_path = root / f"{slug}_kernel_stats.csv"
    counter_path = root / f"{slug}_counter_collection.csv"

    stats_rows = selected_kernel_rows(stats_path, include, min_calls)
    total_ns = sum(float(r["TotalDurationNs"]) for r in stats_rows)
    calls = sum(int(r["Calls"]) for r in stats_rows)

    counter_rows = selected_counter_rows(counter_path, include)
    by_counter = {}
    for row in counter_rows:
        name = row["Counter_Name"]
        value = float(row["Counter_Value"])
        duration = max(
            0.0,
            float(row["End_Timestamp"]) - float(row["Start_Timestamp"]),
        )
        by_counter.setdefault(name, []).append((value, duration))

    fetch_kib = sum(v for v, _ in by_counter.get("FetchSize", [])) / ITERS
    if fetch_kib == 0:
        rd32 = sum(v for v, _ in by_counter.get("GL2C_EA_RDREQ_32B_sum", []))
        rd64 = sum(v for v, _ in by_counter.get("GL2C_EA_RDREQ_64B_sum", []))
        rd128 = sum(v for v, _ in by_counter.get("GL2C_EA_RDREQ_128B_sum", []))
        fetch_kib = (rd32 * 32.0 + rd64 * 64.0 + rd128 * 128.0) / ITERS / 1024.0
    write_req = sum(v for v, _ in by_counter.get("GL2C_EA_WRREQ_64B_sum", [])) / ITERS
    write_kib = write_req * 64.0 / 1024.0
    lds_util = weighted_average(by_counter.get("LdsUtil", []))
    lds_inst = sum(v for v, _ in by_counter.get("SQ_INSTS_LDS", [])) / ITERS
    lds_bank_conflict = sum(v for v, _ in by_counter.get("SQC_LDS_BANK_CONFLICT", [])) / ITERS
    occ_pct = weighted_average(by_counter.get("OccupancyPercent", []))
    mean_occ = weighted_average(by_counter.get("MeanOccupancyPerCU", []))
    waves = sum(v for v, _ in by_counter.get("SQ_WAVES_sum", [])) / ITERS
    grbm_count = sum(v for v, _ in by_counter.get("GRBM_COUNT", []))
    grbm_active = sum(v for v, _ in by_counter.get("GRBM_GUI_ACTIVE", []))
    gpu_active = 100.0 * grbm_active / grbm_count if grbm_count else 0.0

    return {
        "time_us": total_ns / ITERS / 1000.0,
        "calls": calls / ITERS,
        "fetch_kib": fetch_kib,
        "write_kib": write_kib,
        "lds_util": lds_util,
        "lds_inst": lds_inst,
        "lds_bank_conflict": lds_bank_conflict,
        "occ_pct": occ_pct,
        "mean_occ": mean_occ,
        "waves": waves,
        "gpu_active": gpu_active,
        "kernels": len(stats_rows),
    }


def fmt(value, digits=3):
    if abs(value) < 0.0005:
        return "0"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        default="rocprof_results_kernel_fusion",
        help="directory containing rocprofv3 CSV files",
    )
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for block, variant, slug, include, theoretical_bytes, min_calls in WORKLOADS:
        data = summarize_one(root, slug, include, min_calls)
        data["theoretical_kib"] = theoretical_bytes / 1024.0 if theoretical_bytes else None
        rows.append((block, variant, data))

    headers = [
        "Block",
        "Variant",
        "Kernel time / iter (us)",
        "Kernel calls / iter",
        "Global read FetchSize (KiB)",
        "Global write est. (KiB)",
        "Min semantic IO (KiB)",
        "LDS inst / iter",
        "LDS bank conflict / iter",
        "SQ waves / iter",
        "GPU active %",
        "Mean occ / CU",
    ]
    table = ["| " + " | ".join(headers) + " |"]
    table.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for block, variant, data in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    block,
                    variant,
                    fmt(data["time_us"]),
                    fmt(data["calls"], 2),
                    fmt(data["fetch_kib"]),
                    fmt(data["write_kib"]),
                    fmt(data["theoretical_kib"]) if data["theoretical_kib"] is not None else "n/a",
                    fmt(data["lds_inst"]),
                    fmt(data["lds_bank_conflict"]),
                    fmt(data["waves"]),
                    fmt(data["gpu_active"]),
                    fmt(data["mean_occ"]),
                ]
            )
            + " |"
        )

    print("\n".join(table))
    print()
    print(
        "Notes: FetchSize uses rocprofv3 FetchSize when present, otherwise "
        "GL2C_EA_RDREQ_{32,64,128}B_sum. Write bytes are estimated as "
        "GL2C_EA_WRREQ_64B_sum * 64. Min semantic IO is an analytical lower "
        "bound for fused/custom kernels and K2 cache reads; PyTorch unfused "
        "pipelines have additional intermediate traffic. Time is summed across "
        f"selected kernels and divided by {ITERS} iterations."
    )


if __name__ == "__main__":
    main()
