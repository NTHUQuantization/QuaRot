"""Benchmark the unified HIP fused kernels (run on ROCm after installation)."""

import argparse
import time

import torch

from quarot import _HIP


def timed(label, fn, iterations):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    print(f"{label}: {(time.perf_counter() - start) * 1e6 / iterations:.2f} us")


def hadamard_reference(x):
    y = x.float()
    width = y.size(-1)
    stride = 1
    while stride < width:
        y = y.reshape(*y.shape[:-1], -1, stride * 2)
        left, right = y[..., :stride].clone(), y[..., stride:].clone()
        y[..., :stride], y[..., stride:] = left + right, left - right
        y = y.reshape_as(x)
        stride <<= 1
    return (y / width**0.5).half()


def attention_reference(attention):
    rotated = hadamard_reference(attention.transpose(-1, -2).contiguous()).transpose(-1, -2)
    flat = rotated.reshape(rotated.size(0), rotated.size(1), -1)
    scale = (flat.abs().amax(dim=-1, keepdim=True) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
    return torch.round(flat / scale).clamp(-8, 7).to(torch.int8)


def ffn_reference(gate, up):
    value = torch.nn.functional.silu(gate) * up
    value = hadamard_reference(value.reshape(*value.shape[:-1], -1, 256)).reshape_as(value)
    scale = (value.reshape(*value.shape[:-1], -1, 256).abs().amax(-1) / 7).half().clamp_min(torch.finfo(torch.float16).tiny)
    return torch.round(value / scale.repeat_interleave(256, -1)).clamp(-8, 7).to(torch.int8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--ffn-width", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if torch.version.hip is None:
        raise RuntimeError("this benchmark requires a ROCm/HIP PyTorch build")
    attention = torch.randn(args.batch, 1, args.heads, args.head_dim, device="cuda", dtype=torch.float16)
    gate = torch.randn(args.batch, args.ffn_width, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    timed("reference attention output", lambda: attention_reference(attention), args.iterations)
    timed("fused attention output (Hadamard + INT4)", lambda: _HIP.fused_attention_hadamard_quant(attention, args.heads), args.iterations)
    timed("reference FFN", lambda: ffn_reference(gate, up), args.iterations)
    timed("fused FFN (SiLU*up + grouped H256 + per-group INT4)",
          lambda: _HIP.fused_ffn_silu_hadamard_quant(gate, up),
          args.iterations)


if __name__ == "__main__":
    main()
