#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "attention_fusion"))
sys.path.append(os.path.join(ROOT, "ffn_fusion"))

import flashinfer_test._HIP as flashinfer_hip
import ffn_fusion_hip
from attention_fusion import (
    allocate_quantized_kv_cache,
    append_quantized_kv_decode,
    make_uniform_paged_kv_metadata,
    quantize_attention_output,
)


@dataclass
class Inputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor


def seeded_half(shape, scale):
    n = 1
    for dim in shape:
        n *= dim
    x = torch.arange(n, dtype=torch.float32).mul_(scale).sin_().reshape(shape)
    return x.half().cuda()


def hadamard(x):
    y = x.clone()
    size = y.size(-1)
    stride = 1
    while stride < size:
        y = y.reshape(*y.shape[:-1], -1, stride * 2)
        a = y[..., :, :stride].clone()
        b = y[..., :, stride:].clone()
        y[..., :, :stride] = a + b
        y[..., :, stride:] = a - b
        y = y.reshape(*x.shape)
        stride <<= 1
    return y / math.sqrt(size)


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


def quantize_s4(x):
    scale = (x.float().abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q = torch.round(x.float() / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()
    param = torch.stack((scale, scale * 8.0), dim=-1).half()
    return packed, param


def quantize_grouped(x, group_size):
    grouped = x.float().reshape(-1, x.size(-1) // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q = torch.round(grouped / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).reshape(*x.shape[:-1], x.size(-1) // 2)
    scale = scale.reshape(*x.shape[:-1], x.size(-1) // group_size)
    return packed.contiguous(), scale.half()


def dequant_paged_i4_to_f16(kv_i4, param_i4):
    lo = ((kv_i4 & 0x0F).to(torch.int16) - 8).float()
    hi = (((kv_i4 >> 4) & 0x0F).to(torch.int16) - 8).float()
    q = torch.empty((*kv_i4.shape[:-1], kv_i4.shape[-1] * 2), device=kv_i4.device, dtype=torch.float32)
    q[..., 0::2] = lo
    q[..., 1::2] = hi
    return (q * param_i4[..., 0].float().unsqueeze(-1)).half().contiguous()


def append_kv_unfused(inputs, metadata, kv_data, kv_param, rope):
    seq_len = (metadata.pages_per_batch - 1) * metadata.page_size + metadata.last_page_offset[0].item()
    k_in = apply_rope_k(inputs.k, seq_len - 1) if rope else inputs.k
    k_packed, k_param = quantize_s4(hadamard(k_in))
    v_packed, v_param = quantize_s4(hadamard(inputs.v))
    entry = metadata.last_page_offset[0].item() - 1
    page_ids = metadata.indices[metadata.indptr[:-1] + (seq_len - 1) // metadata.page_size]
    kv_data[page_ids, 0, 0, :, entry, :] = k_packed
    kv_data[page_ids, 0, 1, :, entry, :] = v_packed
    kv_param[page_ids, 0, 0, :, entry, :] = k_param
    kv_param[page_ids, 0, 1, :, entry, :] = v_param


def decode_i4(inputs, metadata, kv_data, kv_param, heads, page_size, batch):
    out = torch.empty_like(inputs.q)
    flashinfer_hip.batch_decode_i4(
        out, inputs.q, kv_data, kv_param, metadata.indptr, metadata.indices,
        metadata.last_page_offset, 1, 0, heads, page_size, batch,
    )
    return out


def decode_unfused_i4_to_f16(inputs, metadata, kv_data, kv_param, heads, page_size, batch):
    kv_f16 = dequant_paged_i4_to_f16(kv_data, kv_param)
    param_f16 = torch.empty((*kv_param.shape[:-1], 2), device=kv_param.device, dtype=torch.float16)
    out = torch.empty_like(inputs.q)
    flashinfer_hip.batch_decode_f16(
        out, inputs.q, kv_f16, param_f16, metadata.indptr, metadata.indices,
        metadata.last_page_offset, 1, 0, heads, page_size, batch,
    )
    return out


def k3_unfused(attn):
    return quantize_grouped(hadamard(attn.reshape(attn.size(0), 16, 256)).reshape(attn.size(0), 4096), 256)


def ffn_unfused(gate, up, group_size):
    x = F.silu(gate.float()) * up.float()
    return quantize_grouped(hadamard(x.reshape(-1, x.size(-1) // group_size, group_size)).reshape_as(x), group_size)


def make_inputs(batch, heads, ffn_hidden):
    return Inputs(
        q=seeded_half((batch, heads, 128), 0.011),
        k=seeded_half((batch, heads, 128), 0.013),
        v=seeded_half((batch, heads, 128), 0.017),
        gate=seeded_half((batch, ffn_hidden), 0.019),
        up=seeded_half((batch, ffn_hidden), 0.023),
    )


def make_variant_fn(name, inputs, metadata, heads, page_size, batch, ffn_group, rope) -> Callable[[], Tuple]:
    kv_data, kv_param = allocate_quantized_kv_cache(metadata, 1, heads, inputs.q.device)

    def append_fused():
        append_quantized_kv_decode(
            inputs.k, inputs.v, metadata, kv_data, kv_param,
            num_layers=1, layer_idx=0, apply_rope_to_k=rope,
        )

    def append_unfused():
        append_kv_unfused(inputs, metadata, kv_data, kv_param, rope)

    def run_unfused_all():
        append_unfused()
        attn = decode_unfused_i4_to_f16(inputs, metadata, kv_data, kv_param, heads, page_size, batch)
        attn_pack = k3_unfused(attn.reshape(batch, heads * 128))
        ffn_pack = ffn_unfused(inputs.gate, inputs.up, ffn_group)
        return attn, attn_pack, ffn_pack

    def run_k1_fused():
        append_fused()
        attn = decode_unfused_i4_to_f16(inputs, metadata, kv_data, kv_param, heads, page_size, batch)
        attn_pack = k3_unfused(attn.reshape(batch, heads * 128))
        ffn_pack = ffn_unfused(inputs.gate, inputs.up, ffn_group)
        return attn, attn_pack, ffn_pack

    def run_k1_k2_fused():
        append_fused()
        attn = decode_i4(inputs, metadata, kv_data, kv_param, heads, page_size, batch)
        attn_pack = k3_unfused(attn.reshape(batch, heads * 128))
        ffn_pack = ffn_unfused(inputs.gate, inputs.up, ffn_group)
        return attn, attn_pack, ffn_pack

    def run_attention_fused():
        append_fused()
        attn = decode_i4(inputs, metadata, kv_data, kv_param, heads, page_size, batch)
        attn_pack = quantize_attention_output(attn.reshape(batch, heads * 128))
        ffn_pack = ffn_unfused(inputs.gate, inputs.up, ffn_group)
        return attn, attn_pack, ffn_pack

    def run_full_fused():
        append_fused()
        attn = decode_i4(inputs, metadata, kv_data, kv_param, heads, page_size, batch)
        attn_pack = quantize_attention_output(attn.reshape(batch, heads * 128))
        ffn_pack = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(inputs.gate, inputs.up, ffn_group)
        return attn, attn_pack, ffn_pack

    variants = {
        "unfused_all": run_unfused_all,
        "k1_fused": run_k1_fused,
        "k1_k2_fused": run_k1_k2_fused,
        "attention_fused": run_attention_fused,
        "full_fused": run_full_fused,
    }
    return variants[name]


def time_ms(fn, warmup, iters):
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


def run_bench(args):
    inputs = make_inputs(args.batch, args.heads, args.ffn_hidden)
    metadata = make_uniform_paged_kv_metadata(args.batch, 1, args.page_size, inputs.q.device)
    rows: List[Dict] = []
    for name in ["unfused_all", "k1_fused", "k1_k2_fused", "attention_fused", "full_fused"]:
        fn = make_variant_fn(name, inputs, metadata, args.heads, args.page_size, args.batch, args.ffn_group, args.rope)
        ms = time_ms(fn, args.warmup, args.iters)
        rows.append({"variant": name, "ms": ms, "speedup_vs_unfused": rows[0]["ms"] / ms if rows else 1.0})

    print(json.dumps(rows, indent=2))
    print()
    print("| Variant | Latency (ms) | Speedup vs unfused |")
    print("| --- | ---: | ---: |")
    for row in rows:
        print(f"| {row['variant']} | {row['ms']:.6f} | {row['speedup_vs_unfused']:.2f}x |")


def run_variant(args):
    inputs = make_inputs(args.batch, args.heads, args.ffn_hidden)
    metadata = make_uniform_paged_kv_metadata(args.batch, 1, args.page_size, inputs.q.device)
    fn = make_variant_fn(args.variant, inputs, metadata, args.heads, args.page_size, args.batch, args.ffn_group, args.rope)
    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["unfused_all", "k1_fused", "k1_k2_fused", "attention_fused", "full_fused"])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--ffn-hidden", type=int, default=14336)
    parser.add_argument("--ffn-group", type=int, default=256)
    parser.add_argument("--rope", action="store_true")
    args = parser.parse_args()
    if args.variant:
        run_variant(args)
    else:
        run_bench(args)


if __name__ == "__main__":
    main()
