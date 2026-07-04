#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
from pathlib import Path

import torch

ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "attention_fusion"))

import flashinfer_test._HIP as flashinfer_hip
from attention_fusion import make_uniform_paged_kv_metadata
from single_decoder_layer_benchmark import (
    batch_decode_f16,
    batch_decode_i4,
    error_metrics,
    make_paged_f16,
    make_paged_i4_unfused,
)


HEAD_DIM = 128
PAGE_SIZE = 128


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


def decode_f16_gqa(q, kv, param, metadata, batch, q_heads, kv_heads):
    out = torch.empty_like(q)
    flashinfer_hip.batch_decode_f16_gqa(
        out,
        q.contiguous(),
        kv,
        param,
        metadata.indptr,
        metadata.indices,
        metadata.last_page_offset,
        1,
        0,
        q_heads,
        kv_heads,
        metadata.page_size,
        batch,
    )
    return out


def decode_i4_gqa(q, kv, param, metadata, batch, q_heads, kv_heads):
    out = torch.empty_like(q)
    flashinfer_hip.batch_decode_i4_gqa(
        out,
        q.contiguous(),
        kv,
        param,
        metadata.indptr,
        metadata.indices,
        metadata.last_page_offset,
        1,
        0,
        q_heads,
        kv_heads,
        metadata.page_size,
        batch,
    )
    return out


def make_inputs(batch, seq_len, q_heads, kv_heads, seed):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    q = torch.randn((batch, q_heads, HEAD_DIM), generator=gen, dtype=torch.float32).half().cuda()
    k = torch.randn((batch, seq_len, kv_heads, HEAD_DIM), generator=gen, dtype=torch.float32).half().cuda()
    v = torch.randn((batch, seq_len, kv_heads, HEAD_DIM), generator=gen, dtype=torch.float32).half().cuda()
    return q, k, v


def run_shape(batch, seq_len, q_heads, kv_heads, warmup, iters, seed):
    metadata = make_uniform_paged_kv_metadata(batch, seq_len, PAGE_SIZE, "cuda")
    q, k, v = make_inputs(batch, seq_len, q_heads, kv_heads, seed)
    repeat = q_heads // kv_heads
    k_exp = k.repeat_interleave(repeat, dim=2).contiguous()
    v_exp = v.repeat_interleave(repeat, dim=2).contiguous()

    kv_f16, param_f16 = make_paged_f16(k, v, metadata=metadata)
    kv_f16_exp, param_f16_exp = make_paged_f16(k_exp, v_exp, metadata=metadata)
    ref_f16 = batch_decode_f16(q, kv_f16_exp, param_f16_exp, metadata, batch, q_heads)
    got_f16 = decode_f16_gqa(q, kv_f16, param_f16, metadata, batch, q_heads, kv_heads)

    kv_i4, param_i4 = make_paged_i4_unfused(k, v, rope=False, metadata=metadata)
    kv_i4_exp, param_i4_exp = make_paged_i4_unfused(k_exp, v_exp, rope=False, metadata=metadata)
    ref_i4 = batch_decode_i4(q, kv_i4_exp, param_i4_exp, metadata, batch, q_heads)
    got_i4 = decode_i4_gqa(q, kv_i4, param_i4, metadata, batch, q_heads, kv_heads)
    torch.cuda.synchronize()

    f16_ms = time_call(lambda: decode_f16_gqa(q, kv_f16, param_f16, metadata, batch, q_heads, kv_heads), warmup, iters)
    i4_ms = time_call(lambda: decode_i4_gqa(q, kv_i4, param_i4, metadata, batch, q_heads, kv_heads), warmup, iters)
    f16_exp_ms = time_call(lambda: batch_decode_f16(q, kv_f16_exp, param_f16_exp, metadata, batch, q_heads), warmup, iters)
    i4_exp_ms = time_call(lambda: batch_decode_i4(q, kv_i4_exp, param_i4_exp, metadata, batch, q_heads), warmup, iters)

    rows = []
    for dtype, ref, got, ms, expanded_ms in [
        ("f16", ref_f16, got_f16, f16_ms, f16_exp_ms),
        ("i4", ref_i4, got_i4, i4_ms, i4_exp_ms),
    ]:
        rows.append({
            "batch": batch,
            "seq_len": seq_len,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "dtype": dtype,
            "gqa_latency_ms": ms,
            "expanded_reference_latency_ms": expanded_ms,
            "speedup_vs_expanded": expanded_ms / ms,
            **error_metrics(ref, got),
        })
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with path.open("w") as f:
        f.write("# FlashInfer GQA Decode Benchmark\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for row in rows:
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    val = f"{val:.6g}"
                vals.append(str(val))
            f.write("| " + " | ".join(vals) + " |\n")


def parse_ints(text):
    return [int(x) for x in text.replace(",", " ").split()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="flashinfer_gqa_results")
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--lengths", default="10,128,1024,4096")
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    rows = []
    for batch in parse_ints(args.batches):
        for seq_len in parse_ints(args.lengths):
            rows.extend(run_shape(batch, seq_len, args.q_heads, args.kv_heads, args.warmup, args.iters, args.seed))
    out = Path(args.out_dir)
    write_csv(out / "gqa_decode.csv", rows)
    write_md(out / "summary.md", rows)


if __name__ == "__main__":
    main()
