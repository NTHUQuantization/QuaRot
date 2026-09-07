"""Isolated upstream FHT and fused Qwen3 component benchmarks."""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def measure(fn, warmup, repeats):
    for _ in range(warmup):
        out = fn()
        del out
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
        del out
    return {"median_us": statistics.median(samples), "samples_us": samples,
            "per_transform_ns": statistics.median(samples) * 1000.0}


def iterations(rows, width):
    elements = rows * width
    if elements >= 100_000_000:
        return 1, 3
    if elements >= 10_000_000:
        return 2, 5
    return 5, 10


def upstream_sweep(args):
    import fast_hadamard_transform as hadacore
    import fast_hadamard_transform_op as fast_op
    results = []
    workloads = []
    for batch in args.batches:
        for seq in args.sequences:
            workloads.extend([
                ("attention_h64", batch, seq, batch * seq * 128, 64),
                ("ffn_h256", batch, seq, batch * seq * 100, 256),
            ])
        workloads.append(("decode_kv_h128", batch, 1, batch * 8, 128))
    for name, batch, seq, rows, width in workloads:
        x = torch.empty((rows, width), device="cuda", dtype=torch.float16)
        x.uniform_(-1, 1)
        warmup, repeats = iterations(rows, width)
        for backend, fn in (
            ("hadacore", lambda: hadacore.hadamard_transform(x, 1 / math.sqrt(width))),
            ("fast_op", lambda: fast_op.hadamard_transform(x, 1 / math.sqrt(width))),
        ):
            timing = measure(fn, warmup, repeats)
            timing.update({"backend": backend, "component": name,
                           "batch_size": batch, "sequence_length": seq,
                           "rows": rows, "width": width,
                           "warmup": warmup, "repeats": repeats})
            timing["per_transform_ns"] /= rows
            results.append(timing)
        print(f"upstream {name} B={batch} S={seq} rows={rows} complete", flush=True)
        del x
        torch.cuda.empty_cache()
    return results


def fused_sweep(args):
    import quarot
    backend = quarot._HIP.fht_backend()
    if args.backend and backend != args.backend:
        raise RuntimeError(f"extension is {backend}, requested {args.backend}")
    results = []
    for batch in args.batches:
        for seq in args.sequences:
            attention = torch.empty((batch, seq, 64, 128), device="cuda", dtype=torch.float16)
            attention.uniform_(-1, 1)
            warmup, repeats = iterations(batch * seq, 8192)
            timing = measure(lambda: quarot._HIP.fused_attention_hadamard_quant(attention, 64), warmup, repeats)
            timing.update({"backend": backend, "component": "fused_attention_h64_quant",
                           "batch_size": batch, "sequence_length": seq,
                           "rows": batch * seq, "width": 8192,
                           "warmup": warmup, "repeats": repeats})
            timing["per_transform_ns"] /= batch * seq * 128
            results.append(timing)
            del attention
            gate = torch.empty((batch * seq, 25600), device="cuda", dtype=torch.float16)
            up = torch.empty_like(gate)
            gate.uniform_(-1, 1); up.uniform_(-1, 1)
            warmup, repeats = iterations(batch * seq, 25600)
            timing = measure(lambda: quarot._HIP.fused_ffn_silu_hadamard_quant(gate, up), warmup, repeats)
            timing.update({"backend": backend, "component": "fused_ffn_silu_h256_quant",
                           "batch_size": batch, "sequence_length": seq,
                           "rows": batch * seq, "width": 25600,
                           "warmup": warmup, "repeats": repeats})
            timing["per_transform_ns"] /= batch * seq * 100
            results.append(timing)
            print(f"fused {backend} B={batch} S={seq} complete", flush=True)
            del gate, up
            torch.cuda.empty_cache()
    return results



def fused_k1_sweep(args):
    import quarot
    backend = quarot._HIP.fht_backend()
    if args.backend and backend != args.backend:
        raise RuntimeError(f"extension is {backend}, requested {args.backend}")
    results = []
    query_heads, kv_heads, head_dim = 64, 8, 128
    layers, page_size = 64, 32
    for batch in args.batches:
        query = torch.empty((batch, 1, query_heads, head_dim), device="cuda", dtype=torch.float16)
        key = torch.empty((batch, 1, kv_heads, head_dim), device="cuda", dtype=torch.float16)
        value = torch.empty_like(key)
        query.uniform_(-1, 1); key.uniform_(-1, 1); value.uniform_(-1, 1)
        angles = torch.randn((batch, head_dim), device="cuda")
        cos, sin = angles.cos().half(), angles.sin().half()
        data = torch.empty((batch, layers, 2, kv_heads, page_size, head_dim // 2),
                           device="cuda", dtype=torch.uint8)
        params = torch.empty((batch, layers, 2, kv_heads, page_size, 2),
                             device="cuda", dtype=torch.float16)
        indptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
        indices = torch.arange(batch, device="cuda", dtype=torch.int32)
        last = torch.ones(batch, device="cuda", dtype=torch.int32)
        def run():
            return quarot._HIP.fused_rope_append_kv_i4(
                query, key, value, cos, sin, data, params, indptr, indices,
                last, layers, 0, page_size)
        timing = measure(run, 20, 100)
        timing.update({"backend": backend,
                       "component": "fused_kv_rope_h128_quant_append",
                       "batch_size": batch, "sequence_length": 1,
                       "rows": batch * kv_heads, "width": head_dim,
                       "warmup": 20, "repeats": 100,
                       "query_heads": query_heads, "kv_heads": kv_heads})
        timing["per_transform_ns"] /= batch * kv_heads
        results.append(timing)
        print(f"fused K1 {backend} B={batch} complete", flush=True)
    return results

def parse_csv(value):
    return [int(x) for x in value.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("upstream", "fused", "fused_k1"), required=True)
    p.add_argument("--backend", choices=("hadacore", "fast_op", "naive"))
    p.add_argument("--batches", type=parse_csv, default=[1, 2])
    p.add_argument("--sequences", type=parse_csv, default=[1, 128, 512, 2048, 4096])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    if args.mode == "upstream":
        rows = upstream_sweep(args)
    elif args.mode == "fused":
        rows = fused_sweep(args)
    else:
        rows = fused_k1_sweep(args)
    payload = {"mode": args.mode, "configuration": vars(args), "rows": rows,
               "environment": {"torch": torch.__version__, "hip": torch.version.hip,
                               "gpu": torch.cuda.get_device_properties(0).gcnArchName}}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")

if __name__ == "__main__":
    main()
