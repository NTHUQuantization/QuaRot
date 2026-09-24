"""Reproducible gfx1201 microbenchmarks for Qwen3-32B kernel dimensions."""

import argparse
import gc
import json
import os
import platform
import statistics
from pathlib import Path

import torch


def measure(fn, warmup, iterations, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        starts = [torch.cuda.Event(enable_timing=True)
                  for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True)
                for _ in range(iterations)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        samples.extend(
            start.elapsed_time(end) * 1000.0
            for start, end in zip(starts, ends))
    ordered = sorted(samples)
    return {
        "count": len(samples),
        "median_us": statistics.median(samples),
        "mean_us": statistics.fmean(samples),
        "min_us": min(samples),
        "p10_us": ordered[round((len(ordered) - 1) * 0.10)],
        "p90_us": ordered[round((len(ordered) - 1) * 0.90)],
    }


def packed(rows, width):
    return torch.randint(
        0, 256, (rows, width // 2), device="cuda", dtype=torch.uint8)


def grouped_case(hip, rows, outputs, width, scale_groups):
    activation = packed(rows, width)
    weight = hip.prepack_b(packed(outputs, width))
    activation_scales = torch.rand(
        rows, scale_groups, device="cuda", dtype=torch.float16)
    weight_scales = torch.rand(
        outputs, device="cuda", dtype=torch.float16)

    def run():
        return hip.matmul_bpre_grouped_scale(
            activation, weight, activation_scales, weight_scales,
            outputs, width)

    return run


def multi_case(hip, rows, outputs, width):
    activation = packed(rows, width)
    activation_scales = torch.rand(
        rows, 1, device="cuda", dtype=torch.float16)
    weights = [hip.prepack_b(packed(output, width)) for output in outputs]
    scales = [torch.rand(output, device="cuda", dtype=torch.float16)
              for output in outputs]
    weight_backing = torch.cat(weights).contiguous()
    scale_backing = torch.cat(scales).contiguous()
    weight_views, scale_views = [], []
    weight_offset = scale_offset = 0
    for weight, scale in zip(weights, scales):
        weight_views.append(weight_backing.narrow(
            0, weight_offset, weight.numel()))
        scale_views.append(scale_backing.narrow(
            0, scale_offset, scale.numel()))
        weight_offset += weight.numel()
        scale_offset += scale.numel()
    has_third = len(outputs) == 3

    def run():
        return hip.matmul_bpre_multi_scale(
            activation, activation_scales,
            weight_views[0], scale_views[0],
            weight_views[1], scale_views[1],
            weight_views[2] if has_third else None,
            scale_views[2] if has_third else None,
            outputs[0], outputs[1], outputs[2] if has_third else 0,
            width)

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouped-nwaves", choices=(1, 2, 4, 8),
                        type=int, default=4)
    parser.add_argument("--multi-nwaves", choices=(1, 2, 4, 8),
                        type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ["QUAROT_QWEN3_32B_GROUPED_NWAVES"] = str(
        args.grouped_nwaves)
    os.environ["QUAROT_QWEN3_32B_MULTI_NWAVES"] = str(
        args.multi_nwaves)
    from quarot import _HIP

    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU required")
    torch.manual_seed(20260826)

    specs = [
        ("o_proj_m1", grouped_case,
         (1, 5120, 8192, 1)),
        ("o_proj_m16", grouped_case,
         (16, 5120, 8192, 1)),
        ("down_proj_m1", grouped_case,
         (1, 5120, 25600, 100)),
        ("qkv_m1", multi_case,
         (1, (8192, 1024, 1024), 5120)),
        ("qkv_m16", multi_case,
         (16, (8192, 1024, 1024), 5120)),
        ("gate_up_m1", multi_case,
         (1, (25600, 25600), 5120)),
    ]
    measurements = []
    for name, factory, factory_args in specs:
        run = factory(_HIP, *factory_args)
        result = measure(
            run, args.warmup, args.iterations, args.repeats)
        measurements.append({"name": name, **result})
        del run
        gc.collect()
        torch.cuda.empty_cache()

    attention = torch.randn(
        1, 1, 64, 128, device="cuda", dtype=torch.float16)
    measurements.append({
        "name": "attention_h64_d128_m1",
        **measure(lambda: _HIP.fused_attention_hadamard_quant(
            attention, 64), args.warmup, args.iterations, args.repeats),
    })
    gate = torch.randn(
        1, 25600, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    measurements.append({
        "name": "ffn_grouped_h256_w25600_m1",
        **measure(lambda: _HIP.fused_ffn_silu_hadamard_quant_grouped256(
            gate, up), args.warmup, args.iterations, args.repeats),
    })
    hidden = torch.randn(
        1, 1, 5120, device="cuda", dtype=torch.float16)
    measurements.append({
        "name": "rmsnorm_quant_w5120_m1",
        **measure(lambda: _HIP.fused_rmsnorm_quant_i4(
            hidden, 1e-6, 0.9), args.warmup, args.iterations, args.repeats),
    })

    result = {
        "environment": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "rocm": torch.version.hip,
            "gpu": torch.cuda.get_device_name(0) or "AMD gfx1201",
            "extension": _HIP.__file__,
        },
        "configuration": vars(args) | {
            "output": None if args.output is None else str(args.output),
            "timing": "HIP events; synchronized after each repeat",
        },
        "measurements": measurements,
    }
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
