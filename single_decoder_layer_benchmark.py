#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


D_MODEL = 4096
HEAD_DIM = 128
NUM_HEADS = D_MODEL // HEAD_DIM
PAGE_SIZE = 128


@dataclass
class LayerWeights:
    rms_attn: torch.Tensor
    rms_ffn: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


@dataclass
class ShapeConfig:
    batch: int
    seq_len: int
    ffn_hidden: int


def seeded_half(shape, seed, scale=0.02):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).half().cuda()


def make_weights(ffn_hidden: int, seed: int) -> LayerWeights:
    return LayerWeights(
        rms_attn=seeded_half((D_MODEL,), seed + 1, 1.0),
        rms_ffn=seeded_half((D_MODEL,), seed + 2, 1.0),
        wq=seeded_half((D_MODEL, D_MODEL), seed + 3),
        wk=seeded_half((D_MODEL, D_MODEL), seed + 4),
        wv=seeded_half((D_MODEL, D_MODEL), seed + 5),
        wo=seeded_half((D_MODEL, D_MODEL), seed + 6),
        w_gate=seeded_half((D_MODEL, ffn_hidden), seed + 7),
        w_up=seeded_half((D_MODEL, ffn_hidden), seed + 8),
        w_down=seeded_half((ffn_hidden, D_MODEL), seed + 9),
    )


def rms_norm(x, weight, eps=1e-6):
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)).half() * weight


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


def apply_rope_k(k, positions, theta=10000.0):
    # k: [B, L, H, D] or [B, H, D]
    original_shape = k.shape
    if k.dim() == 3:
        x = k.float().reshape(k.size(0), 1, k.size(1), -1, 2)
        pos = positions.reshape(k.size(0), 1, 1)
    else:
        x = k.float().reshape(k.size(0), k.size(1), k.size(2), -1, 2)
        pos = positions.reshape(1, -1, 1)
    pair_idx = torch.arange(x.size(-2), device=k.device, dtype=torch.float32)
    freq = (1.0 / theta) ** (2.0 * pair_idx / HEAD_DIM)
    angle = pos.float().unsqueeze(-1) * freq
    c = torch.cos(angle)
    s = torch.sin(angle)
    y0 = x[..., 0] * c - x[..., 1] * s
    y1 = x[..., 0] * s + x[..., 1] * c
    return torch.stack((y0, y1), dim=-1).reshape(original_shape).half()


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
    return packed.contiguous(), scale.reshape(*x.shape[:-1], x.size(-1) // group_size).half()


def dequant_s4(packed, param):
    lo = ((packed & 0x0F).to(torch.int16) - 8).float()
    hi = (((packed >> 4) & 0x0F).to(torch.int16) - 8).float()
    q = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), device=packed.device, dtype=torch.float32)
    q[..., 0::2] = lo
    q[..., 1::2] = hi
    return (q * param[..., 0].float().unsqueeze(-1)).half()


def dequant_grouped(packed, scales, group_size):
    lo = ((packed & 0x0F).to(torch.int16) - 8).float()
    hi = (((packed >> 4) & 0x0F).to(torch.int16) - 8).float()
    q = torch.empty((*packed.shape[:-1], packed.shape[-1] * 2), device=packed.device, dtype=torch.float32)
    q[..., 0::2] = lo
    q[..., 1::2] = hi
    return (q.reshape(*q.shape[:-1], -1, group_size) * scales.float().unsqueeze(-1)).reshape(
        *q.shape
    ).half()


def make_paged_f16(k, v, page_size=PAGE_SIZE, metadata=None, total_seq_len=None):
    # k/v: [B, L, H, D]
    batch, seq_len, heads, head_dim = k.shape
    if metadata is not None:
        pages_per_batch = metadata.pages_per_batch
        total_pages = metadata.total_pages
    else:
        pages_per_batch = ((total_seq_len or seq_len) + page_size - 1) // page_size
        total_pages = batch * pages_per_batch
    kv = torch.zeros((total_pages, 1, 2, heads, page_size, head_dim), device=k.device, dtype=torch.float16)
    param = torch.zeros((total_pages, 1, 2, heads, page_size, 2), device=k.device, dtype=torch.float16)
    for b in range(batch):
        for p in range(pages_per_batch):
            begin = p * page_size
            end = min(begin + page_size, seq_len)
            page = b * pages_per_batch + p
            kv[page, 0, 0, :, : end - begin] = k[b, begin:end].transpose(0, 1)
            kv[page, 0, 1, :, : end - begin] = v[b, begin:end].transpose(0, 1)
    return kv, param


def make_paged_i4_unfused(k, v, page_size=PAGE_SIZE, rope=True, metadata=None, total_seq_len=None):
    # k/v: [B, L, H, D]
    batch, seq_len, heads, _ = k.shape
    if metadata is not None:
        pages_per_batch = metadata.pages_per_batch
        total_pages = metadata.total_pages
    else:
        pages_per_batch = ((total_seq_len or seq_len) + page_size - 1) // page_size
        total_pages = batch * pages_per_batch
    kv = torch.zeros((total_pages, 1, 2, heads, page_size, HEAD_DIM // 2), device=k.device, dtype=torch.uint8)
    param = torch.zeros((total_pages, 1, 2, heads, page_size, 2), device=k.device, dtype=torch.float16)
    if seq_len == 0:
        return kv, param
    positions = torch.arange(seq_len, device=k.device, dtype=torch.int32)
    k_in = apply_rope_k(k, positions) if rope else k
    k_rot = hadamard(k_in)
    v_rot = hadamard(v)
    k_packed, k_param = quantize_s4(k_rot)
    v_packed, v_param = quantize_s4(v_rot)
    for b in range(batch):
        for p in range(pages_per_batch):
            begin = p * page_size
            end = min(begin + page_size, seq_len)
            page = b * pages_per_batch + p
            kv[page, 0, 0, :, : end - begin] = k_packed[b, begin:end].transpose(0, 1)
            kv[page, 0, 1, :, : end - begin] = v_packed[b, begin:end].transpose(0, 1)
            param[page, 0, 0, :, : end - begin] = k_param[b, begin:end].transpose(0, 1)
            param[page, 0, 1, :, : end - begin] = v_param[b, begin:end].transpose(0, 1)
    return kv, param


def make_paged_i4_with_fused_last(k, v, metadata, page_size=PAGE_SIZE):
    kv, param = make_paged_i4_unfused(
        k[:, :-1],
        v[:, :-1],
        page_size,
        metadata=metadata,
        total_seq_len=k.size(1),
    )
    append_quantized_kv_decode(
        k[:, -1],
        v[:, -1],
        metadata,
        kv,
        param,
        num_layers=1,
        layer_idx=0,
        apply_rope_to_k=True,
    )
    return kv, param


def make_metadata(batch, seq_len, page_size=PAGE_SIZE):
    return make_uniform_paged_kv_metadata(batch, seq_len, page_size, "cuda")


def batch_decode_f16(q, kv, param, metadata, batch, heads, page_size=PAGE_SIZE):
    out = torch.empty_like(q)
    flashinfer_hip.batch_decode_f16(
        out, q, kv, param, metadata.indptr, metadata.indices, metadata.last_page_offset,
        1, 0, heads, page_size, batch,
    )
    return out


def batch_decode_i4(q, kv, param, metadata, batch, heads, page_size=PAGE_SIZE):
    out = torch.empty_like(q)
    flashinfer_hip.batch_decode_i4(
        out, q, kv, param, metadata.indptr, metadata.indices, metadata.last_page_offset,
        1, 0, heads, page_size, batch,
    )
    return out


def dequant_paged_i4(kv_i4, param_i4):
    return dequant_s4(kv_i4, param_i4).contiguous()


class SingleDecoderLayerHarness:
    def __init__(self, shape: ShapeConfig, seed: int = 123):
        self.shape = shape
        self.weights = make_weights(shape.ffn_hidden, seed)
        self.hidden_cur = seeded_half((shape.batch, D_MODEL), seed + 20, 0.1)
        self.metadata = make_metadata(shape.batch, shape.seq_len)
        self.prev_k_raw = seeded_half((shape.batch, max(shape.seq_len - 1, 0), NUM_HEADS, HEAD_DIM), seed + 21, 0.02)
        self.prev_v = seeded_half((shape.batch, max(shape.seq_len - 1, 0), NUM_HEADS, HEAD_DIM), seed + 22, 0.02)
        self.kv_f16, self.param_f16 = self.make_fp16_cache()
        self.kv_i4, self.param_i4 = make_paged_i4_unfused(
            self.prev_k_raw,
            self.prev_v,
            metadata=self.metadata,
            total_seq_len=shape.seq_len,
        )

    def make_fp16_cache(self):
        if self.shape.seq_len == 1:
            prev_k = self.prev_k_raw
        else:
            positions = torch.arange(self.shape.seq_len - 1, device=self.hidden_cur.device, dtype=torch.int32)
            prev_k = apply_rope_k(self.prev_k_raw, positions)
        return make_paged_f16(prev_k, self.prev_v, page_size=PAGE_SIZE, metadata=self.metadata, total_seq_len=self.shape.seq_len)

    def projections(self):
        w = self.weights
        x_cur = rms_norm(self.hidden_cur, w.rms_attn)
        q = (x_cur @ w.wq).reshape(self.shape.batch, NUM_HEADS, HEAD_DIM).contiguous()
        k = (x_cur @ w.wk).reshape(self.shape.batch, NUM_HEADS, HEAD_DIM).contiguous()
        v = (x_cur @ w.wv).reshape(self.shape.batch, NUM_HEADS, HEAD_DIM).contiguous()
        return x_cur, q, k, v

    def append_fp16_current(self, k_cur, v_cur):
        slot = (self.shape.seq_len - 1) % PAGE_SIZE
        page_ids = self.metadata.indices[self.metadata.indptr[:-1] + (self.shape.seq_len - 1) // PAGE_SIZE]
        positions = torch.full((self.shape.batch,), self.shape.seq_len - 1, device=k_cur.device, dtype=torch.int32)
        k_rope = apply_rope_k(k_cur, positions)
        self.kv_f16[page_ids, 0, 0, :, slot, :] = k_rope
        self.kv_f16[page_ids, 0, 1, :, slot, :] = v_cur

    def append_i4_unfused_current(self, k_cur, v_cur):
        slot = (self.shape.seq_len - 1) % PAGE_SIZE
        page_ids = self.metadata.indices[self.metadata.indptr[:-1] + (self.shape.seq_len - 1) // PAGE_SIZE]
        positions = torch.full((self.shape.batch,), self.shape.seq_len - 1, device=k_cur.device, dtype=torch.int32)
        k_rope = apply_rope_k(k_cur, positions)
        k_packed, k_param = quantize_s4(hadamard(k_rope))
        v_packed, v_param = quantize_s4(hadamard(v_cur))
        self.kv_i4[page_ids, 0, 0, :, slot, :] = k_packed
        self.kv_i4[page_ids, 0, 1, :, slot, :] = v_packed
        self.param_i4[page_ids, 0, 0, :, slot, :] = k_param
        self.param_i4[page_ids, 0, 1, :, slot, :] = v_param

    def ffn_inputs(self, attn_hidden):
        x = rms_norm(attn_hidden, self.weights.rms_ffn)
        gate = (x @ self.weights.w_gate).contiguous()
        up = (x @ self.weights.w_up).contiguous()
        return gate, up

    def run(self, variant: str):
        batch, seq_len = self.shape.batch, self.shape.seq_len
        w = self.weights
        x_cur, q, k_cur, v_cur = self.projections()

        if variant == "fp16_baseline":
            self.append_fp16_current(k_cur, v_cur)
            attn = batch_decode_f16(q, self.kv_f16, self.param_f16, self.metadata, batch, NUM_HEADS)
            o_in = attn.reshape(batch, D_MODEL)
            ffn_gate_source = x_cur + (o_in @ w.wo)
            gate, up = self.ffn_inputs(ffn_gate_source)
            ffn_intermediate = F.silu(gate.float()) * up.float()
            down = (ffn_intermediate.half() @ w.w_down)
            return x_cur + (o_in @ w.wo) + down

        use_k1 = variant in {
            "k1_fused",
            "k1_k2_fused",
            "attention_fused",
            "attention_fused_current",
            "attention_fused_hadacore256",
            "full_fused",
            "fused_quarot",
            "full_fused_current",
            "full_fused_hadacore256_ffn",
            "full_fused_hadacore256_k3_ffn",
        }
        use_k2 = variant in {
            "k1_k2_fused",
            "attention_fused",
            "attention_fused_current",
            "attention_fused_hadacore256",
            "full_fused",
            "fused_quarot",
            "full_fused_current",
            "full_fused_hadacore256_ffn",
            "full_fused_hadacore256_k3_ffn",
        }
        use_k3 = variant in {
            "attention_fused",
            "attention_fused_current",
            "attention_fused_hadacore256",
            "full_fused",
            "fused_quarot",
            "full_fused_current",
            "full_fused_hadacore256_ffn",
            "full_fused_hadacore256_k3_ffn",
        }
        use_ffn = variant in {
            "full_fused",
            "fused_quarot",
            "full_fused_current",
            "full_fused_hadacore256_ffn",
            "full_fused_hadacore256_k3_ffn",
        }
        use_k3_hadacore = variant in {"attention_fused_hadacore256", "full_fused_hadacore256_k3_ffn"}
        use_ffn_hadacore = variant in {"full_fused_hadacore256_ffn", "full_fused_hadacore256_k3_ffn"}

        if use_k1:
            append_quantized_kv_decode(
                k_cur, v_cur, self.metadata, self.kv_i4, self.param_i4,
                num_layers=1, layer_idx=0, apply_rope_to_k=True,
            )
        else:
            self.append_i4_unfused_current(k_cur, v_cur)

        if use_k2:
            attn = batch_decode_i4(q, self.kv_i4, self.param_i4, self.metadata, batch, NUM_HEADS)
        else:
            kv_dequant = dequant_paged_i4(self.kv_i4, self.param_i4)
            param_f16 = torch.zeros((*self.param_i4.shape[:-1], 2), device=q.device, dtype=torch.float16)
            attn = batch_decode_f16(q, kv_dequant, param_f16, self.metadata, batch, NUM_HEADS)

        attn_2d = attn.reshape(batch, D_MODEL)
        if use_k3:
            packed, scales = quantize_attention_output(
                attn_2d,
                backend="hadacore256" if use_k3_hadacore else "current",
            )
            o_in = dequant_grouped(packed, scales, 256)
        else:
            packed, scales = quantize_grouped(hadamard(attn_2d.reshape(batch, 16, 256)).reshape(batch, D_MODEL), 256)
            o_in = dequant_grouped(packed, scales, 256)

        ffn_gate_source = x_cur + (o_in @ w.wo)
        gate, up = self.ffn_inputs(ffn_gate_source)
        if use_ffn:
            if use_ffn_hadacore:
                ffn_packed, ffn_scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant_hadacore256(gate, up)
            else:
                ffn_packed, ffn_scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(gate, up, 256)
        else:
            ffn_packed, ffn_scales = quantize_grouped(
                hadamard((F.silu(gate.float()) * up.float()).reshape(batch, -1, 256)).reshape(batch, self.shape.ffn_hidden),
                256,
            )
        ffn_in = dequant_grouped(ffn_packed, ffn_scales, 256)
        down = ffn_in @ w.w_down
        return ffn_gate_source + down


VARIANTS = [
    "fp16_baseline",
    "quarot_unfused",
    "k1_fused",
    "k1_k2_fused",
    "attention_fused",
    "attention_fused_hadacore256",
    "fused_quarot",
    "full_fused_hadacore256_ffn",
    "full_fused_hadacore256_k3_ffn",
]

ALIASES = {
    "quarot_unfused": "quarot_unfused",
    "unfused_all": "quarot_unfused",
    "full_fused": "fused_quarot",
    "unfused_INT4": "quarot_unfused",
    "attention_fused_current": "attention_fused",
    "full_fused_current": "fused_quarot",
}


def canonical_variant(name):
    return ALIASES.get(name, name)


def error_metrics(ref, out):
    diff = (out.float() - ref.float()).abs()
    rel = diff / ref.float().abs().clamp_min(1e-6)
    return {
        "max_error": diff.max().item(),
        "mean_error": diff.mean().item(),
        "mean_relative_error": rel.mean().item(),
    }


def tolerance_result(row, args):
    if row["reference"] == "fp16_baseline" and row["variant"] != "fp16_baseline":
        return "NA_quantization_drift"
    passed = row["max_error"] <= args.max_error_tol and row["mean_error"] <= args.mean_error_tol
    return "PASS" if passed else "FAIL"


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


def parse_ints(text):
    return [int(x) for x in text.replace(",", " ").split()]


def shape_iter(args) -> Iterable[ShapeConfig]:
    for batch in parse_ints(args.batches):
        for seq_len in parse_ints(args.lengths):
            for ffn_hidden in parse_ints(args.ffn_hidden):
                yield ShapeConfig(batch, seq_len, ffn_hidden)


def bench(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    correctness_rows: List[Dict] = []
    latency_rows: List[Dict] = []

    for shape in shape_iter(args):
        harness = SingleDecoderLayerHarness(shape, args.seed)
        ref_fp16 = harness.run("fp16_baseline")
        ref_quarot = harness.run("quarot_unfused")
        torch.cuda.synchronize()
        for variant in ["fp16_baseline", "quarot_unfused", "fused_quarot"]:
            out = harness.run(variant)
            torch.cuda.synchronize()
            row_fp16 = {
                "batch": shape.batch,
                "seq_len": shape.seq_len,
                "ffn_hidden": shape.ffn_hidden,
                "variant": variant,
                "reference": "fp16_baseline",
                **error_metrics(ref_fp16, out),
            }
            row_fp16["tolerance"] = f"max<={args.max_error_tol},mean<={args.mean_error_tol}"
            row_fp16["result"] = tolerance_result(row_fp16, args)
            correctness_rows.append(row_fp16)
            if variant != "fp16_baseline":
                row_quarot = {
                    "batch": shape.batch,
                    "seq_len": shape.seq_len,
                    "ffn_hidden": shape.ffn_hidden,
                    "variant": variant,
                    "reference": "quarot_unfused",
                    **error_metrics(ref_quarot, out),
                }
                row_quarot["tolerance"] = f"max<={args.max_error_tol},mean<={args.mean_error_tol}"
                row_quarot["result"] = tolerance_result(row_quarot, args)
                correctness_rows.append(row_quarot)

        for variant in ["fp16_baseline", "quarot_unfused", "fused_quarot"]:
            ms = time_call(lambda v=variant: harness.run(v), args.warmup, args.iters)
            latency_rows.append({
                "batch": shape.batch,
                "seq_len": shape.seq_len,
                "ffn_hidden": shape.ffn_hidden,
                "variant": variant,
                "latency_ms": ms,
            })

    write_csv(out_dir / "correctness.csv", correctness_rows)
    write_csv(out_dir / "latency.csv", latency_rows)
    write_markdown_summary(out_dir / "summary.md", correctness_rows, latency_rows)


def ablation(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    variants = [
        "quarot_unfused",
        "k1_fused",
        "k1_k2_fused",
        "attention_fused_current",
        "attention_fused_hadacore256",
        "full_fused_current",
        "full_fused_hadacore256_ffn",
        "full_fused_hadacore256_k3_ffn",
    ]
    for shape in shape_iter(args):
        harness = SingleDecoderLayerHarness(shape, args.seed)
        baseline = None
        for variant in variants:
            ms = time_call(lambda v=variant: harness.run(v), args.warmup, args.iters)
            if baseline is None:
                baseline = ms
            rows.append({
                "batch": shape.batch,
                "seq_len": shape.seq_len,
                "ffn_hidden": shape.ffn_hidden,
                "variant": variant,
                "latency_ms": ms,
                "speedup_vs_unfused_INT4": baseline / ms,
            })
    write_csv(out_dir / "ablation.csv", rows)
    write_ablation_markdown(out_dir / "ablation.md", rows)


def run_variant(args):
    shape = ShapeConfig(args.batch, args.seq_len, args.ffn_hidden_single)
    harness = SingleDecoderLayerHarness(shape, args.seed)
    variant = canonical_variant(args.variant)
    for _ in range(args.iters):
        harness.run(variant)
    torch.cuda.synchronize()


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(f, rows, columns):
    f.write("| " + " | ".join(columns) + " |\n")
    f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        values = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.6g}"
            values.append(str(val))
        f.write("| " + " | ".join(values) + " |\n")


def write_markdown_summary(path, correctness_rows, latency_rows):
    with path.open("w") as f:
        f.write("# Single Decoder Layer Benchmark Summary\n\n")
        f.write("## Correctness\n\n")
        write_markdown_table(
            f,
            correctness_rows,
            ["batch", "seq_len", "ffn_hidden", "variant", "reference", "max_error", "mean_error", "mean_relative_error", "tolerance", "result"],
        )
        f.write("\n## Latency\n\n")
        write_markdown_table(
            f,
            latency_rows,
            ["batch", "seq_len", "ffn_hidden", "variant", "latency_ms"],
        )


def write_ablation_markdown(path, rows):
    with path.open("w") as f:
        f.write("# Single Decoder Layer Ablation\n\n")
        write_markdown_table(
            f,
            rows,
            ["batch", "seq_len", "ffn_hidden", "variant", "latency_ms", "speedup_vs_unfused_INT4"],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bench", "ablation", "variant"], default="bench")
    parser.add_argument("--out-dir", default="single_decoder_layer_results")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--lengths", default="10,128,1024,4096")
    parser.add_argument("--ffn-hidden", default="11008,14336")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-error-tol", type=float, default=1.25)
    parser.add_argument("--mean-error-tol", type=float, default=0.25)
    parser.add_argument("--variant", default="fused_quarot")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--ffn-hidden-single", type=int, default=14336)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    if args.mode == "bench":
        bench(args)
    elif args.mode == "ablation":
        ablation(args)
    else:
        run_variant(args)


if __name__ == "__main__":
    main()
