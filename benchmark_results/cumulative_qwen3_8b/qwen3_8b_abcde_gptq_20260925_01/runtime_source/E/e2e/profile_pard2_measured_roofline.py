#!/usr/bin/env python3
"""Measure representative PARD2 kernels for an empirical roofline.

GL2C byte counters return zero on the tested gfx1201/ROCm 7.2 stack.  This
benchmark therefore combines measured cold-cache HIP-event latency with exact
algorithmic tensor bytes.  It never labels those bytes as PM-counter DRAM
traffic.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

import quarot


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "pard2_optimization_results" / "measured_roofline_summary.json"
FLUSH_BYTES = 512 * 1024 * 1024
WARMUPS = 5
SAMPLES = 30


def timed_cold(fn):
    flush = timed_cold.flush
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUPS):
        flush.add_(1)
        fn()
        torch.cuda.synchronize()
    samples = []
    for _ in range(SAMPLES):
        flush.add_(1)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    ordered = sorted(samples)
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "cv": statistics.stdev(samples) / statistics.mean(samples),
        "p10_us": ordered[2],
        "p90_us": ordered[-3],
    }


def roof_point(logical_m, k, n, latency_us, algorithmic_bytes):
    operations = 2 * logical_m * k * n
    seconds = latency_us * 1.0e-6
    return {
        "logical_m": logical_m,
        "operations": operations,
        "algorithmic_bytes": algorithmic_bytes,
        "operational_intensity_op_per_byte": operations / algorithmic_bytes,
        "measured_throughput_top_s": operations / seconds / 1.0e12,
        "algorithmic_bandwidth_gb_s": algorithmic_bytes / seconds / 1.0e9,
        "bandwidth_roof_utilization": algorithmic_bytes / seconds / 640.0e9,
    }


def main():
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("requires a ROCm/HIP GPU")
    torch.manual_seed(20260825)
    timed_cold.flush = torch.zeros(
        FLUSH_BYTES, device="cuda", dtype=torch.uint8
    )

    k, n, physical_m = 4096, 12288, 16
    activation = torch.randint(
        0, 256, (physical_m, k // 2), device="cuda", dtype=torch.uint8
    )
    weight = torch.randint(
        0, 256, (n, k // 2), device="cuda", dtype=torch.uint8
    )
    prepacked = quarot._HIP.prepack_b(weight)
    w4_timing = timed_cold(
        lambda: quarot._HIP.matmul_bpre(activation, prepacked, n, k)
    )
    w4_bytes = n * k // 2 + physical_m * k // 2 + physical_m * n * 4

    pm, pk, pn = 15, 16384, 1024
    features = torch.randn(pm, pk, device="cuda", dtype=torch.bfloat16)
    projection = torch.randn(pk, pn, device="cuda", dtype=torch.bfloat16)
    projected = torch.empty(pm, pn, device="cuda", dtype=torch.bfloat16)
    bf16_timing = timed_cold(
        lambda: torch.mm(features, projection, out=projected)
    )
    bf16_bytes = (pm * pk + pk * pn + pm * pn) * 2

    rows, width = 16, 4096
    norm_input = torch.randn(
        1, rows, width, device="cuda", dtype=torch.float16
    )
    norm_timing = timed_cold(
        lambda: quarot._HIP.rms_norm_quant_i4_rows(norm_input, width, 1e-6)
    )
    norm_bytes = rows * width * 2 + rows * width // 2 + rows * 2
    norm_seconds = norm_timing["median_us"] * 1.0e-6

    result = {
        "contract": {
            "gpu": torch.cuda.get_device_name(0),
            "warmups": WARMUPS,
            "samples": SAMPLES,
            "cache_flush_bytes": FLUSH_BYTES,
            "timing": "HIP events",
            "bytes": "exact algorithmic tensor bytes, not PM-counter DRAM bytes",
            "gl2c_counter_status": "unusable: zero for nonzero 8/24 MiB operands",
        },
        "w4a4_4096x12288": {
            "timing": w4_timing,
            "physical_m": physical_m,
            "m1": roof_point(1, k, n, w4_timing["median_us"], w4_bytes),
            "m16": roof_point(16, k, n, w4_timing["median_us"], w4_bytes),
        },
        "bf16_projection_15x16384x1024": {
            "timing": bf16_timing,
            "point": roof_point(pm, pk, pn, bf16_timing["median_us"], bf16_bytes),
        },
        "rms_norm_quant_16x4096": {
            "timing": norm_timing,
            "algorithmic_bytes": norm_bytes,
            "algorithmic_bandwidth_gb_s": norm_bytes / norm_seconds / 1.0e9,
            "roofline_point_omitted": "mixed reduction, FP and integer op semantics",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
