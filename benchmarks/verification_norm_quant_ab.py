"""Reproducible primitive A/B; run only when the shared GPU is idle.

PYTHONPATH=$PWD python benchmarks/verification_norm_quant_ab.py \
    --output verification_optimization_20260924/norm_primitive.json

These primitive timings are not target verification latency or PARD-2 TPS.
The model-level A/B harness measures those separately.
"""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
import quarot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 15, 16])
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--clip-ratio", type=float, default=0.9)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--inner", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("requires the project's ROCm GPU environment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260924)
    rng = random.Random(20260924)
    quantizer = quarot.nn.Quantizer(args.clip_ratio)
    result = {
        "scope": "primitive-only; no verification latency or PARD-2 TPS inference",
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "hip": torch.version.hip, "width": args.width,
        "clip_ratio": args.clip_ratio, "samples": args.samples,
        "inner": args.inner,
        "extension_sha256": hashlib.sha256(
            Path(quarot._HIP.__file__).read_bytes()).hexdigest(),
        "shapes": [],
    }
    for rows in args.rows:
        value = torch.randn(1, rows, args.width, device="cuda", dtype=torch.float16)
        residual = torch.randn_like(value)

        def baseline(x):
            packed = quantizer(quarot._HIP.rms_norm_rows(x, args.width, 1e-6))
            return packed.quantized_x, packed.scales_x

        def fused(x):
            return quarot._HIP.rms_norm_quant_i4_rows_clipped(
                x, args.width, 1e-6, args.clip_ratio)

        methods = {
            "norm_unfused": lambda: baseline(value),
            "norm_fused": lambda: fused(value),
            "residual_unfused": lambda: baseline(value + residual),
            "residual_add_plus_fused_norm": lambda: fused(value + residual),
            "residual_fused": lambda: quarot._HIP.residual_rms_norm_quant_i4_rows(
                value, residual, args.width, 1e-6, args.clip_ratio),
        }
        for name, fn in methods.items():
            actual = fn()
            expected = baseline(value if name.startswith("norm_") else value + residual)
            assert torch.equal(actual[0], expected[0]), name
            assert torch.equal(actual[1], expected[1]), name
            if name == "residual_fused":
                assert torch.equal(actual[2], value + residual)
            for _ in range(20):
                fn()
        torch.cuda.synchronize()
        samples = {name: [] for name in methods}
        for _ in range(args.samples):
            order = list(methods)
            rng.shuffle(order)
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.inner):
                    methods[name]()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) / args.inner)
        record = {"rows": rows, "bit_exact": True, "methods": {}}
        for name, fn in methods.items():
            trace_path = args.output.parent / f"norm_M{rows}_{name}.trace.json"
            with torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(trace_path))
            events = json.loads(trace_path.read_text())["traceEvents"]
            kernels = [event for event in events if event.get("cat") == "kernel"]
            record["methods"][name] = {
                "event_median_ms": statistics.median(samples[name]),
                "event_samples_ms": samples[name],
                "kernel_launches": len(kernels),
                "kernel_names": [event["name"] for event in kernels],
                "trace": str(trace_path),
            }
        result["shapes"].append(record)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"rows": rows, "methods": {
            name: {k: item[k] for k in ("event_median_ms", "kernel_launches")}
            for name, item in record["methods"].items()}}), flush=True)


if __name__ == "__main__":
    main()
