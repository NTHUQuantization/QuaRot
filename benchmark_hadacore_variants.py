#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "attention_fusion"))
sys.path.append(os.path.join(ROOT, "ffn_fusion"))

import ffn_fusion_hip
from quarot_attention_fusion import (
    hadamard_reference,
    quantize_attention_output,
    quantize_attention_output_hadacore256,
    quantize_attention_output_hadacore4096_experimental,
    quantize_grouped_reference,
)


def seeded_half(shape, seed, scale=0.01):
    torch.manual_seed(seed)
    return torch.randn(shape, device="cuda", dtype=torch.float16) * scale


def ffn_reference(gate, up, group_size):
    x = F.silu(gate.float()) * up.float()
    rotated = hadamard_reference(x.reshape(gate.size(0), -1, group_size)).reshape_as(x)
    return quantize_grouped_reference(rotated, group_size)


def k3_block_reference(out):
    return quantize_grouped_reference(hadamard_reference(out.reshape(out.size(0), 16, 256)).reshape_as(out), 256)


def k3_full_reference(out):
    return quantize_grouped_reference(hadamard_reference(out), 256)


def time_call(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def mismatch_metrics(packed, scales, ref_packed, ref_scales):
    mismatch = (packed != ref_packed).sum().item()
    total = ref_packed.numel()
    scale_diff = (scales.float() - ref_scales.float()).abs()
    return {
        "packed_mismatch": mismatch,
        "packed_total": total,
        "packed_mismatch_rate": mismatch / total,
        "scale_max_error": scale_diff.max().item(),
        "scale_mean_error": scale_diff.mean().item(),
    }


def bench_ffn(rows, cols, warmup, iters):
    gate = seeded_half((rows, cols), 1000 + rows + cols, 1.0)
    up = seeded_half((rows, cols), 2000 + rows + cols, 1.0)
    ref_packed, ref_scales = ffn_reference(gate, up, 256)

    def unfused():
        ffn_reference(gate, up, 256)

    variants = {
        "ffn_unfused_INT4": unfused,
        "ffn_current_fused": lambda: ffn_fusion_hip.fused_ffn_silu_hadamard_quant(gate, up, 256),
        "ffn_hadacore256_fused": lambda: ffn_fusion_hip.fused_ffn_silu_hadamard_quant_hadacore256(gate, up),
    }
    baseline_ms = None
    rows_out = []
    for name, fn in variants.items():
        ms = time_call(fn, warmup, iters)
        if name == "ffn_unfused_INT4":
            baseline_ms = ms
            packed, scales = ref_packed, ref_scales
        else:
            packed, scales = fn()
            torch.cuda.synchronize()
        metrics = mismatch_metrics(packed, scales, ref_packed, ref_scales)
        rows_out.append({
            "block": "FFN",
            "variant": name,
            "rows": rows,
            "cols": cols,
            "latency_ms": ms,
            "speedup_vs_unfused_INT4": baseline_ms / ms,
            **metrics,
        })
    return rows_out


def bench_k3(rows, warmup, iters):
    out = seeded_half((rows, 4096), 3000 + rows, 1.0)
    block_ref_packed, block_ref_scales = k3_block_reference(out)
    full_ref_packed, full_ref_scales = k3_full_reference(out)

    def unfused():
        k3_block_reference(out)

    variants = {
        "k3_unfused_INT4": (unfused, block_ref_packed, block_ref_scales),
        "k3_current_fused": (lambda: quantize_attention_output(out), block_ref_packed, block_ref_scales),
        "k3_hadacore256_fused": (lambda: quantize_attention_output_hadacore256(out), block_ref_packed, block_ref_scales),
        "k3_hadacore4096_experimental": (
            lambda: quantize_attention_output_hadacore4096_experimental(out),
            full_ref_packed,
            full_ref_scales,
        ),
    }
    baseline_ms = None
    rows_out = []
    for name, (fn, ref_packed, ref_scales) in variants.items():
        ms = time_call(fn, warmup, iters)
        if name == "k3_unfused_INT4":
            baseline_ms = ms
            packed, scales = ref_packed, ref_scales
        else:
            packed, scales = fn()
            torch.cuda.synchronize()
        metrics = mismatch_metrics(packed, scales, ref_packed, ref_scales)
        rows_out.append({
            "block": "K3",
            "variant": name,
            "rows": rows,
            "cols": 4096,
            "latency_ms": ms,
            "speedup_vs_unfused_INT4": baseline_ms / ms,
            **metrics,
        })
    return rows_out


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k3-rows", default="1,2,4,8")
    parser.add_argument("--ffn-rows", default="1,2,4,8")
    parser.add_argument("--ffn-cols", default="11008,14336")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", default="hadacore_variant_results/component_benchmark.csv")
    args = parser.parse_args()

    rows = []
    for k3_rows in [int(x) for x in args.k3_rows.replace(",", " ").split()]:
        rows.extend(bench_k3(k3_rows, args.warmup, args.iters))
    for ffn_rows in [int(x) for x in args.ffn_rows.replace(",", " ").split()]:
        for cols in [int(x) for x in args.ffn_cols.replace(",", " ").split()]:
            rows.extend(bench_ffn(ffn_rows, cols, args.warmup, args.iters))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
