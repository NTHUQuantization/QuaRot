"""Synchronized fused-vs-unfused HIP microbenchmarks with JSON output."""

import argparse
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path

import torch
from quarot import _HIP


def hadamard(x):
    y = x.float()
    width = y.size(-1)
    stride = 1
    while stride < width:
        y = y.reshape(*y.shape[:-1], -1, stride * 2)
        left = y[..., :stride].clone()
        right = y[..., stride:].clone()
        y[..., :stride] = left + right
        y[..., stride:] = left - right
        y = y.reshape_as(x)
        stride <<= 1
    return y / math.sqrt(width)


def pack_s4(x, scale):
    q = torch.round(x / scale).clamp(-8, 7).to(torch.int8)
    return ((q[..., 0::2].to(torch.uint8) & 15) |
            ((q[..., 1::2].to(torch.uint8) & 15) << 4))


def attention_unfused(x):
    b, s, h, d = x.shape
    y = hadamard(x.transpose(-1, -2).contiguous()).transpose(-1, -2).reshape(b, s, h * d)
    scale = (y.abs().amax(-1, keepdim=True) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
    return pack_s4(y, scale), scale


def ffn_unfused(gate, up):
    y = torch.nn.functional.silu(gate.float()) * up.float()
    y = hadamard(y.reshape(*y.shape[:-1], -1, 256)).reshape_as(y)
    scale = (y.reshape(*y.shape[:-1], -1, 256).abs().amax(-1) / 7).half()
    scale.clamp_min_(torch.finfo(torch.float16).tiny)
    return pack_s4(y, scale.repeat_interleave(256, -1)), scale


def measure(fn, warmup, iterations, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        samples.extend(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))
    ordered = sorted(samples)
    percentile = lambda p: ordered[round((len(ordered) - 1) * p)]
    return {
        "samples_us": samples,
        "count": len(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "stddev_us": statistics.pstdev(samples),
        "p10_us": percentile(0.10),
        "p90_us": percentile(0.90),
        "min_us": min(samples),
    }


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args()
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU required")
    torch.manual_seed(20260802)
    cases = []
    specs = [
        ("attention_decode", (1, 1, 32, 128)),
        ("attention_prefill_16", (1, 16, 32, 128)),
        ("attention_prefill_128", (1, 128, 32, 128)),
    ]
    for name, shape in specs:
        x = torch.randn(*shape, device="cuda", dtype=torch.float16)
        fused = measure(lambda: _HIP.fused_attention_hadamard_quant(x, shape[2]), args.warmup, args.iterations, args.repeats)
        unfused = measure(lambda: attention_unfused(x), args.warmup, args.iterations, args.repeats)
        speedup = unfused["median_us"] / fused["median_us"]
        cases.append({"name": name, "shape": list(shape), "dtype": "float16", "fused": fused,
                      "unfused": unfused, "speedup": speedup,
                      "latency_reduction_percent": (1.0 - 1.0 / speedup) * 100.0})
    # Physical widths cover common Llama/Qwen FFNs. 29696 is the padded form
    # of Qwen's 29568-wide MLP and exercises the universal padding contract.
    ffn_specs = [
        ("ffn_decode", 1, 11008),
        ("ffn_decode", 1, 14336),
        ("ffn_decode", 1, 22016),
        ("ffn_decode", 1, 28672),
        ("ffn_decode_padded", 1, 29696),
        ("ffn_prefill_16", 16, 14336),
        ("ffn_prefill_128", 128, 14336),
    ]
    for name, batch, width in ffn_specs:
        gate = torch.randn(batch, width, device="cuda", dtype=torch.float16)
        up = torch.randn_like(gate)
        fused = measure(
            lambda: _HIP.fused_ffn_silu_hadamard_quant(
                gate, up),
            args.warmup, args.iterations, args.repeats)
        unfused = measure(lambda: ffn_unfused(gate, up), args.warmup, args.iterations, args.repeats)
        speedup = unfused["median_us"] / fused["median_us"]
        cases.append({"name": name, "shape": [batch, width], "dtype": "float16", "fused": fused,
                      "unfused": unfused, "speedup": speedup,
                      "latency_reduction_percent": (1.0 - 1.0 / speedup) * 100.0})
    result = {
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "torch": torch.__version__, "rocm": torch.version.hip,
                        "gpu": torch.cuda.get_device_name(0) or "AMD gfx1201", "extension": _HIP.__file__},
        "git": {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD")},
        "configuration": vars(args) | {"output": str(args.output), "timing": "HIP events; synchronized per repeat"},
        "measurements": cases,
        "skipped": [
            {"case": "BF16", "reason": "fused API explicitly supports FP16 only"},
            {"case": "end_to_end_model", "reason": "no local model path or weights configured"},
            {"case": "KV-length latency sweep", "reason": "audited fused output kernels do not depend on KV length"},
        ],
        "failures": [],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        print("{} {}: fused={:.3f} us unfused={:.3f} us speedup={:.3f}x".format(
            case["name"], case["shape"], case["fused"]["median_us"],
            case["unfused"]["median_us"], case["speedup"]
        ))


if __name__ == "__main__":
    main()
