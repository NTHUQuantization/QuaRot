import argparse
import math
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn.functional as F

import ffn_fusion_hip


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HADACORE_DIR = os.path.join(REPO_ROOT, "fast-hadamard-for-hip", "hadacore")
if HADACORE_DIR not in sys.path:
    sys.path.append(HADACORE_DIR)

try:
    from hadacore_for_hip import hadacore
except ImportError:
    hadacore = None


def block_hadamard_torch(x, group_size):
    y = x.reshape(-1, x.size(-1) // group_size, group_size).clone()
    stride = 1
    while stride < group_size:
        y = y.reshape(*y.shape[:-1], -1, stride * 2)
        a = y[..., :, :stride].clone()
        b = y[..., :, stride:].clone()
        y[..., :, :stride] = a + b
        y[..., :, stride:] = a - b
        y = y.reshape(-1, x.size(-1) // group_size, group_size)
        stride <<= 1
    return (y / math.sqrt(group_size)).reshape_as(x)


def quantize_s4(x, group_size):
    grouped = x.float().reshape(-1, x.size(-1) // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q = torch.round(grouped / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).reshape(*x.shape[:-1], x.size(-1) // 2)
    return packed.contiguous(), scale.reshape(*x.shape[:-1], x.size(-1) // group_size).half()


def ffn_unfused_torch(gate, up, group_size):
    x = F.silu(gate.float()) * up.float()
    h = block_hadamard_torch(x, group_size)
    return quantize_s4(h, group_size)


def ffn_hadacore_pipeline(gate, up, group_size):
    if hadacore is None:
        raise RuntimeError("hadacore_for_hip is not importable")
    x = (F.silu(gate.float()) * up.float()).to(torch.float16)
    grouped = x.reshape(-1, group_size).contiguous()
    h = hadacore(grouped, 1.0 / math.sqrt(group_size)).reshape_as(x)
    return quantize_s4(h, group_size)


def time_call(fn, iters, warmup):
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


def profile_case(rows, cols, group_size, iters, warmup, stage_iters):
    torch.manual_seed(0)
    gate = torch.randn((rows, cols), device="cuda", dtype=torch.float16)
    up = torch.randn((rows, cols), device="cuda", dtype=torch.float16)

    fused_packed, fused_scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(gate, up, group_size)
    torch.cuda.synchronize()
    torch_packed, torch_scales = ffn_unfused_torch(gate, up, group_size)
    torch.cuda.synchronize()

    mismatch = (fused_packed != torch_packed).sum().item()
    scale_err = (fused_scales.float() - torch_scales.float()).abs().max().item()

    fused_ms = time_call(
        lambda: ffn_fusion_hip.fused_ffn_silu_hadamard_quant(gate, up, group_size),
        iters,
        warmup,
    )
    torch_ms = time_call(lambda: ffn_unfused_torch(gate, up, group_size), iters, warmup)

    hadacore_ms = None
    hadacore_mismatch = None
    if hadacore is not None:
        hc_packed, _ = ffn_hadacore_pipeline(gate, up, group_size)
        torch.cuda.synchronize()
        hadacore_mismatch = (hc_packed != torch_packed).sum().item()
        hadacore_ms = time_call(lambda: ffn_hadacore_pipeline(gate, up, group_size), iters, warmup)

    silu_ms = time_call(lambda: F.silu(gate.float()) * up.float(), stage_iters, warmup)
    x = (F.silu(gate.float()) * up.float()).to(torch.float16)
    torch_had_ms = time_call(lambda: block_hadamard_torch(x.float(), group_size), stage_iters, 5)
    quant_ms = time_call(lambda: quantize_s4(x, group_size), stage_iters, warmup)
    hc_had_ms = None
    if hadacore is not None:
        grouped = x.reshape(-1, group_size).contiguous()
        hc_had_ms = time_call(lambda: hadacore(grouped, 1.0 / math.sqrt(group_size)), stage_iters, warmup)

    print(
        f"shape=({rows},{cols}) group={group_size} "
        f"fused_ms={fused_ms:.6f} torch_unfused_ms={torch_ms:.6f} "
        f"speedup_vs_torch={torch_ms / fused_ms:.2f}x "
        f"mismatch={mismatch} scale_maxerr={scale_err:.6g}"
    )
    if hadacore_ms is not None:
        print(
            f"  hadacore_pipeline_ms={hadacore_ms:.6f} "
            f"speedup_vs_hadacore_pipeline={hadacore_ms / fused_ms:.2f}x "
            f"hadacore_mismatch_vs_torch={hadacore_mismatch}"
        )
    print(
        "  stages_ms "
        f"silu_mul={silu_ms:.6f} torch_block_had={torch_had_ms:.6f} "
        f"quant={quant_ms:.6f}"
        + (f" hadacore_block_had={hc_had_ms:.6f}" if hc_had_ms is not None else "")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", default="11008,14336")
    parser.add_argument("--rows", default="1,8,32,128")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--stage-iters", type=int, default=100)
    args = parser.parse_args()

    for cols in [int(x) for x in args.cols.replace(",", " ").split()]:
        for rows in [int(x) for x in args.rows.replace(",", " ").split()]:
            profile_case(rows, cols, args.group_size, args.iters, args.warmup, args.stage_iters)


if __name__ == "__main__":
    main()
