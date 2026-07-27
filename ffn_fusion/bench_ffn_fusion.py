import argparse
import math

import torch
import torch.nn.functional as F

import ffn_fusion_hip


def hadamard_inplace(x):
    size = x.size(-1)
    y = x.clone()
    stride = 1
    while stride < size:
        y = y.reshape(*y.shape[:-1], -1, stride * 2)
        a = y[..., :, :stride].clone()
        b = y[..., :, stride:].clone()
        y[..., :, :stride] = a + b
        y[..., :, stride:] = a - b
        y = y.reshape(*x.shape)
        stride *= 2
    return y


def reference(gate, up, group_size):
    x = F.silu(gate.float()) * up.float()
    grouped = x.reshape(-1, x.size(-1) // group_size, group_size)
    rotated = hadamard_inplace(grouped) / math.sqrt(group_size)
    scale = (rotated.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q = torch.round(rotated / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).reshape(*gate.shape[:-1], gate.size(-1) // 2)
    return packed.contiguous(), scale.reshape(*gate.shape[:-1], gate.size(-1) // group_size).half(), q


def unpack_s4(packed):
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def run_backend(gate, up, group_size, backend):
    if backend == "current":
        return ffn_fusion_hip.fused_ffn_silu_hadamard_quant(gate, up, group_size)
    if backend == "hadacore256":
        if group_size != 256:
            raise ValueError("hadacore256 backend requires group_size=256")
        return ffn_fusion_hip.fused_ffn_silu_hadamard_quant_hadacore256(gate, up)
    raise ValueError(f"unknown backend: {backend}")


def bench(rows, cols, group_size, backend, iters, warmup, ref_iters):
    torch.manual_seed(0)
    gate = torch.randn((rows, cols), device="cuda", dtype=torch.float16)
    up = torch.randn((rows, cols), device="cuda", dtype=torch.float16)

    packed, scales = run_backend(gate, up, group_size, backend)
    torch.cuda.synchronize()
    ref_packed, ref_scales, ref_q = reference(gate, up, group_size)
    got_q = unpack_s4(packed).reshape(-1, cols // group_size, group_size)

    packed_mismatches = (packed != ref_packed).sum().item()
    q_maxerr = (got_q.cpu().to(torch.int16) - ref_q.cpu()).abs().max().item()
    scale_maxerr = (scales.float() - ref_scales.float()).abs().max().item()

    for _ in range(warmup):
        run_backend(gate, up, group_size, backend)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_backend(gate, up, group_size, backend)
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / iters

    ref_avg_ms = None
    if ref_iters > 0:
        for _ in range(5):
            reference(gate, up, group_size)
        torch.cuda.synchronize()

        ref_start = torch.cuda.Event(enable_timing=True)
        ref_end = torch.cuda.Event(enable_timing=True)
        ref_start.record()
        for _ in range(ref_iters):
            reference(gate, up, group_size)
        ref_end.record()
        torch.cuda.synchronize()
        ref_avg_ms = ref_start.elapsed_time(ref_end) / ref_iters

    message = (
        f"shape=({rows},{cols}) group={group_size} "
        f"backend={backend} "
        f"packed_mismatch={packed_mismatches} q_maxerr={q_maxerr} "
        f"scale_maxerr={scale_maxerr:.6g} fused_ms={avg_ms:.6f}"
    )
    if ref_avg_ms is not None:
        message += f" torch_ref_ms={ref_avg_ms:.6f} speedup={ref_avg_ms / avg_ms:.2f}x"
    print(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=14336)
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--backend", choices=["current", "hadacore256"], default="current")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--ref-iters", type=int, default=50)
    args = parser.parse_args()
    bench(args.rows, args.cols, args.group_size, args.backend, args.iters, args.warmup, args.ref_iters)


if __name__ == "__main__":
    main()
