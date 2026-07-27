import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "ffn_fusion"))
sys.path.append(os.path.join(ROOT, "attention_fusion"))
sys.path.append(ROOT)
sys.path.append(os.path.abspath(os.path.join(ROOT, "..", "fast-hadamard-for-hip", "hadacore")))

import attention_fusion_hip
import ffn_fusion_hip
import flashinfer_test._HIP as flashinfer_hip
from hadacore_for_hip import hadacore


def seeded_half(shape, scale=0.01):
    n = 1
    for dim in shape:
        n *= dim
    data = torch.arange(n, dtype=torch.float32).mul_(scale).sin_().reshape(shape).half()
    return data.to("cuda")


def hadamard_torch(x):
    y = x.clone()
    n = y.size(-1)
    stride = 1
    while stride < n:
        y = y.reshape(*y.shape[:-1], -1, stride * 2)
        a = y[..., :, :stride].clone()
        b = y[..., :, stride:].clone()
        y[..., :, :stride] = a + b
        y[..., :, stride:] = a - b
        y = y.reshape(*x.shape)
        stride <<= 1
    return y / math.sqrt(n)


def quantize_grouped(x, group):
    grouped = x.float().reshape(-1, x.size(-1) // group, group)
    scale = (grouped.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q = torch.round(grouped / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).reshape(*x.shape[:-1], x.size(-1) // 2)
    return packed.contiguous(), scale.reshape(*x.shape[:-1], x.size(-1) // group).half()


def quantize_s4(x):
    scale = (x.float().abs().amax(dim=-1) / 7.0).clamp_min(1e-6)
    q = torch.round(x.float() / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()
    param = torch.stack([scale, scale * 8.0], dim=-1).half()
    return packed, param


def apply_rope_k(k, pos, theta=10000.0):
    x = k.float().reshape(*k.shape[:-1], -1, 2)
    pair_idx = torch.arange(x.size(-2), device=k.device, dtype=torch.float32)
    freq = (1.0 / theta) ** (2.0 * pair_idx / k.size(-1))
    angle = pos * freq
    c = torch.cos(angle)
    s = torch.sin(angle)
    y0 = x[..., 0] * c - x[..., 1] * s
    y1 = x[..., 0] * s + x[..., 1] * c
    return torch.stack((y0, y1), dim=-1).reshape_as(k).half()


def run_ffn_unfused(iters, rows=1, cols=14336):
    gate = seeded_half((rows, cols), 0.01)
    up = seeded_half((rows, cols), 0.013)
    for _ in range(iters):
        x = F.silu(gate.float()) * up.float()
        h = hadamard_torch(x.reshape(rows, cols // 256, 256)).reshape_as(x)
        quantize_grouped(h, 256)


def run_ffn_fused(iters, rows=1, cols=14336):
    gate = seeded_half((rows, cols), 0.01)
    up = seeded_half((rows, cols), 0.013)
    packed = None
    for _ in range(iters):
        packed = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(gate, up, 256)
    return packed


def run_ffn_hadacore256_fused(iters, rows=1, cols=14336):
    gate = seeded_half((rows, cols), 0.01)
    up = seeded_half((rows, cols), 0.013)
    packed = None
    for _ in range(iters):
        packed = ffn_fusion_hip.fused_ffn_silu_hadamard_quant_hadacore256(gate, up)
    return packed


def run_ffn_hadacore_pipeline_reference(iters, rows=1, cols=14336):
    gate = seeded_half((rows, cols), 0.01)
    up = seeded_half((rows, cols), 0.013)
    for _ in range(iters):
        x = (F.silu(gate.float()) * up.float()).half()
        h = hadacore(x.reshape(-1, 256).contiguous(), 1.0 / math.sqrt(256)).reshape_as(x)
        quantize_grouped(h, 256)


def make_decode_data(seq_len=4096, heads=8, head_dim=128, page_size=128):
    q = seeded_half((1, heads, head_dim), 0.011)
    k = seeded_half((1, heads, seq_len, head_dim), 0.007)
    v = seeded_half((1, heads, seq_len, head_dim), 0.009)
    pages = (seq_len + page_size - 1) // page_size
    indptr = torch.tensor([0, pages], device="cuda", dtype=torch.int32)
    indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    last = torch.tensor([seq_len - (pages - 1) * page_size], device="cuda", dtype=torch.int32)

    kv_f16 = torch.zeros((pages, 1, 2, heads, page_size, head_dim), device="cuda", dtype=torch.float16)
    for p in range(pages):
        b = p * page_size
        e = min(b + page_size, seq_len)
        kv_f16[p, 0, 0, :, : e - b] = k[0, :, b:e]
        kv_f16[p, 0, 1, :, : e - b] = v[0, :, b:e]
    param_f16 = torch.zeros((pages, 1, 2, heads, page_size, 2), device="cuda", dtype=torch.float16)

    kp, kparam = quantize_s4(k)
    vp, vparam = quantize_s4(v)
    kv_i4 = torch.zeros((pages, 1, 2, heads, page_size, head_dim // 2), device="cuda", dtype=torch.uint8)
    param_i4 = torch.zeros((pages, 1, 2, heads, page_size, 2), device="cuda", dtype=torch.float16)
    for p in range(pages):
        b = p * page_size
        e = min(b + page_size, seq_len)
        kv_i4[p, 0, 0, :, : e - b] = kp[0, :, b:e]
        kv_i4[p, 0, 1, :, : e - b] = vp[0, :, b:e]
        param_i4[p, 0, 0, :, : e - b] = kparam[0, :, b:e]
        param_i4[p, 0, 1, :, : e - b] = vparam[0, :, b:e]
    return q, kv_f16, param_f16, kv_i4, param_i4, indptr, indices, last


def dequant_paged_i4_to_f16(kv_i4, param_i4):
    lo = ((kv_i4 & 0x0F).to(torch.int16) - 8).float()
    hi = ((kv_i4 >> 4).to(torch.int16) - 8).float()
    q = torch.empty((*kv_i4.shape[:-1], kv_i4.shape[-1] * 2), device=kv_i4.device, dtype=torch.float32)
    q[..., 0::2] = lo
    q[..., 1::2] = hi
    scale = param_i4[..., 0].float().unsqueeze(-1)
    return (q * scale).half().contiguous()


def run_k2_f16(iters):
    q, kv_f16, param_f16, _, _, indptr, indices, last = make_decode_data()
    out = torch.empty_like(q)
    for _ in range(iters):
        flashinfer_hip.batch_decode_f16(out, q, kv_f16, param_f16, indptr, indices, last, 1, 0, 8, 128, 1)


def run_k2_i4(iters):
    q, _, _, kv_i4, param_i4, indptr, indices, last = make_decode_data()
    out = torch.empty_like(q)
    for _ in range(iters):
        flashinfer_hip.batch_decode_i4(out, q, kv_i4, param_i4, indptr, indices, last, 1, 0, 8, 128, 1)


def run_k2_quarot_unfused(iters):
    q, _, param_f16, kv_i4, param_i4, indptr, indices, last = make_decode_data()
    out = torch.empty_like(q)
    for _ in range(iters):
        kv_dequant = dequant_paged_i4_to_f16(kv_i4, param_i4)
        flashinfer_hip.batch_decode_f16(out, q, kv_dequant, param_f16, indptr, indices, last, 1, 0, 8, 128, 1)


def run_k1(iters, rope):
    key = seeded_half((1, 8, 128), 0.011)
    value = seeded_half((1, 8, 128), 0.017)
    pages = 32
    indptr = torch.tensor([0, pages], device="cuda", dtype=torch.int32)
    indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    last = torch.tensor([128], device="cuda", dtype=torch.int32)
    kv = torch.empty((pages, 1, 2, 8, 128, 64), device="cuda", dtype=torch.uint8)
    param = torch.empty((pages, 1, 2, 8, 128, 2), device="cuda", dtype=torch.float16)
    for _ in range(iters):
        attention_fusion_hip.append_kv_had_quant_inplace(
            key, value, kv, param, indptr, indices, last, 1, 0, 8, 128, 1, rope
        )


def run_k1_unfused(iters, rope):
    key = seeded_half((1, 8, 128), 0.011)
    value = seeded_half((1, 8, 128), 0.017)
    pages = 32
    indptr = torch.tensor([0, pages], device="cuda", dtype=torch.int32)
    indices = torch.arange(pages, device="cuda", dtype=torch.int32)
    last = torch.tensor([128], device="cuda", dtype=torch.int32)
    kv = torch.empty((pages, 1, 2, 8, 128, 64), device="cuda", dtype=torch.uint8)
    param = torch.empty((pages, 1, 2, 8, 128, 2), device="cuda", dtype=torch.float16)
    page = indices[indptr[0] + (last[0] - 1) // 128]
    entry = (last[0] - 1) % 128
    for _ in range(iters):
        k_in = apply_rope_k(key, 127) if rope else key
        k_h = hadamard_torch(k_in)
        v_h = hadamard_torch(value)
        k_packed, k_param = quantize_s4(k_h)
        v_packed, v_param = quantize_s4(v_h)
        kv[page, 0, 0, :, entry, :] = k_packed[0]
        kv[page, 0, 1, :, entry, :] = v_packed[0]
        param[page, 0, 0, :, entry, :] = k_param[0]
        param[page, 0, 1, :, entry, :] = v_param[0]


def run_k3(iters, rows=1):
    out = seeded_half((rows, 4096), 0.011)
    packed = torch.empty((rows, 2048), device="cuda", dtype=torch.uint8)
    scales = torch.empty((rows, 16), device="cuda", dtype=torch.float16)
    for _ in range(iters):
        attention_fusion_hip.output_had_quant_inplace(out, packed, scales)


def run_k3_hadacore256(iters, rows=1):
    out = seeded_half((rows, 4096), 0.011)
    for _ in range(iters):
        attention_fusion_hip.output_had_quant_hadacore256(out)


def run_k3_hadacore4096_experimental(iters, rows=1):
    out = seeded_half((rows, 4096), 0.011)
    for _ in range(iters):
        attention_fusion_hip.output_had_quant_hadacore4096_experimental(out)


def run_k3_unfused(iters, rows=1):
    out = seeded_half((rows, 4096), 0.011)
    for _ in range(iters):
        h = hadamard_torch(out.reshape(rows, 16, 256)).reshape_as(out)
        quantize_grouped(h, 256)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workload")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=14336)
    args = parser.parse_args()
    torch.manual_seed(0)

    workloads = {
        "ffn_unfused": lambda iters: run_ffn_unfused(iters, args.rows, args.cols),
        "ffn_unfused_INT4": lambda iters: run_ffn_unfused(iters, args.rows, args.cols),
        "ffn_hadacore": lambda iters: run_ffn_hadacore_pipeline_reference(iters, args.rows, args.cols),
        "ffn_hadacore_pipeline_reference": lambda iters: run_ffn_hadacore_pipeline_reference(iters, args.rows, args.cols),
        "ffn_fused": lambda iters: run_ffn_fused(iters, args.rows, args.cols),
        "ffn_current_fused": lambda iters: run_ffn_fused(iters, args.rows, args.cols),
        "ffn_hadacore256_fused": lambda iters: run_ffn_hadacore256_fused(iters, args.rows, args.cols),
        "k1_unfused_no_rope": lambda iters: run_k1_unfused(iters, False),
        "k1_unfused_rope": lambda iters: run_k1_unfused(iters, True),
        "k1_no_rope": lambda iters: run_k1(iters, False),
        "k1_rope": lambda iters: run_k1(iters, True),
        "k2_f16": run_k2_f16,
        "k2_quarot_unfused": run_k2_quarot_unfused,
        "k2_i4": run_k2_i4,
        "k3_unfused": lambda iters: run_k3_unfused(iters, args.rows),
        "k3_unfused_INT4": lambda iters: run_k3_unfused(iters, args.rows),
        "k3": lambda iters: run_k3(iters, args.rows),
        "k3_current_fused": lambda iters: run_k3(iters, args.rows),
        "k3_hadacore256_fused": lambda iters: run_k3_hadacore256(iters, args.rows),
        "k3_hadacore4096_experimental": lambda iters: run_k3_hadacore4096_experimental(iters, args.rows),
    }
    workloads[args.workload](args.iters)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
