#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "attention_fusion"))
sys.path.append(str(ROOT / "ffn_fusion"))

import flashinfer_test._HIP as flashinfer_hip
import ffn_fusion_hip
from attention_fusion import append_quantized_kv_decode, make_uniform_paged_kv_metadata, quantize_attention_output
from llama31_quarot.benchmark_full_model import make_inputs
from llama31_quarot.common import (
    RuntimeConfig,
    cuda_event_time_ms,
    error_metrics,
    load_prompts,
    load_tokenizer_and_model,
    logits_metrics,
    summarize,
    write_csv,
)
from single_decoder_layer_benchmark import (
    dequant_grouped,
    dequant_paged_i4,
    hadamard,
    make_paged_f16,
    make_paged_i4_unfused,
    quantize_grouped,
    quantize_s4,
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


PAGE_SIZE = 128
GROUP_SIZE = 256


@dataclass
class PreparedInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    next_ids: torch.Tensor
    hf_past_key_values: object
    prompt_kv: list
    metadata: object
    f16_caches: list
    i4_caches: list


def cache_layer_kv(past_key_values, layer_idx):
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    return past_key_values[layer_idx]


def parse_ints(text):
    return [int(x) for x in text.replace(",", " ").split()]


def decode_gqa_f16(q, kv, param, metadata, batch, q_heads, kv_heads):
    out = torch.empty_like(q)
    flashinfer_hip.batch_decode_f16_gqa(
        out,
        q.contiguous(),
        kv,
        param,
        metadata.indptr,
        metadata.indices,
        metadata.last_page_offset,
        1,
        0,
        q_heads,
        kv_heads,
        metadata.page_size,
        batch,
    )
    return out


def decode_gqa_i4(q, kv, param, metadata, batch, q_heads, kv_heads):
    out = torch.empty_like(q)
    flashinfer_hip.batch_decode_i4_gqa(
        out,
        q.contiguous(),
        kv,
        param,
        metadata.indptr,
        metadata.indices,
        metadata.last_page_offset,
        1,
        0,
        q_heads,
        kv_heads,
        metadata.page_size,
        batch,
    )
    return out


def append_f16_current(kv, k_cur, v_cur, metadata, seq_len):
    slot = (seq_len - 1) % metadata.page_size
    page_ids = metadata.indices[metadata.indptr[:-1] + (seq_len - 1) // metadata.page_size]
    kv[page_ids, 0, 0, :, slot, :] = k_cur
    kv[page_ids, 0, 1, :, slot, :] = v_cur


def append_i4_unfused_current(kv, param, k_cur, v_cur, metadata, seq_len):
    slot = (seq_len - 1) % metadata.page_size
    page_ids = metadata.indices[metadata.indptr[:-1] + (seq_len - 1) // metadata.page_size]
    k_packed, k_param = quantize_s4(hadamard(k_cur))
    v_packed, v_param = quantize_s4(hadamard(v_cur))
    kv[page_ids, 0, 0, :, slot, :] = k_packed
    kv[page_ids, 0, 1, :, slot, :] = v_packed
    param[page_ids, 0, 0, :, slot, :] = k_param
    param[page_ids, 0, 1, :, slot, :] = v_param


@torch.inference_mode()
def prepare_inputs(model, tokenizer, prompt, batch, context_len, device):
    inputs = make_inputs(tokenizer, prompt, batch, context_len, device)
    outputs = model(**inputs, use_cache=True)
    next_ids = inputs["input_ids"][:, -1:]
    total_len = inputs["input_ids"].size(1) + 1
    metadata = make_uniform_paged_kv_metadata(batch, total_len, PAGE_SIZE, device)
    prompt_kv = []
    f16_caches = []
    i4_caches = []
    for layer_idx in range(model.config.num_hidden_layers):
        k, v = cache_layer_kv(outputs.past_key_values, layer_idx)
        # HF cache is [B, kv_heads, L, D]; local paged cache helpers use [B, L, kv_heads, D].
        prompt_k = k.transpose(1, 2).contiguous()
        prompt_v = v.transpose(1, 2).contiguous()
        prompt_kv.append((prompt_k, prompt_v))
        f16_caches.append(make_paged_f16(prompt_k, prompt_v, metadata=metadata))
        i4_caches.append(make_paged_i4_unfused(prompt_k, prompt_v, rope=False, metadata=metadata))
    return PreparedInputs(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        next_ids=next_ids,
        hf_past_key_values=outputs.past_key_values,
        prompt_kv=prompt_kv,
        metadata=metadata,
        f16_caches=f16_caches,
        i4_caches=i4_caches,
    )


def layer_project(layer, hidden, position_embeddings):
    batch, q_len, _ = hidden.shape
    attn = layer.self_attn
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    q = attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    k = attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    v = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1])
    return q.squeeze(2).contiguous(), k.squeeze(2).contiguous(), v.squeeze(2).contiguous()


def make_layer_cache(prompt_k, prompt_v, variant, metadata):
    if variant == "fp16_manual":
        return make_paged_f16(prompt_k, prompt_v, metadata=metadata)
    return make_paged_i4_unfused(prompt_k, prompt_v, rope=False, metadata=metadata)


@torch.inference_mode()
def run_manual_decode(model, prepared: PreparedInputs, variant: str):
    if variant == "fp16_hf":
        attention_mask = torch.ones(
            (prepared.next_ids.size(0), prepared.input_ids.size(1) + 1),
            device=prepared.next_ids.device,
            dtype=torch.long,
        )
        return model(
            input_ids=prepared.next_ids,
            attention_mask=attention_mask,
            past_key_values=prepared.hf_past_key_values,
            use_cache=True,
        ).logits[:, -1]

    cfg = model.config
    batch = prepared.next_ids.size(0)
    prompt_len = prepared.input_ids.size(1)
    total_len = prompt_len + 1
    q_heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    metadata = prepared.metadata

    hidden = model.model.embed_tokens(prepared.next_ids)
    cache_position = torch.full((batch, 1), prompt_len, device=prepared.next_ids.device, dtype=torch.long)
    position_embeddings = model.model.rotary_emb(hidden, cache_position)

    for layer_idx, layer in enumerate(model.model.layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        q, k_cur, v_cur = layer_project(layer, normed, position_embeddings)
        if variant == "fp16_manual":
            kv_cache, kv_param = prepared.f16_caches[layer_idx]
            append_f16_current(kv_cache, k_cur, v_cur, metadata, total_len)
            attn_out = decode_gqa_f16(q, kv_cache, kv_param, metadata, batch, q_heads, kv_heads)
            o_in = attn_out.reshape(batch, 1, cfg.hidden_size)
        elif variant == "quarot_unfused":
            kv_cache, kv_param = prepared.i4_caches[layer_idx]
            append_i4_unfused_current(kv_cache, kv_param, k_cur, v_cur, metadata, total_len)
            kv_dequant = dequant_paged_i4(kv_cache, kv_param)
            param_f16 = torch.zeros((*kv_param.shape[:-1], 2), device=q.device, dtype=torch.float16)
            attn_out = decode_gqa_f16(q, kv_dequant, param_f16, metadata, batch, q_heads, kv_heads)
            packed, scales = quantize_grouped(hadamard(attn_out.reshape(batch, 16, GROUP_SIZE)).reshape(batch, cfg.hidden_size), GROUP_SIZE)
            o_in = dequant_grouped(packed, scales, GROUP_SIZE).reshape(batch, 1, cfg.hidden_size)
        elif variant == "fused_quarot":
            kv_cache, kv_param = prepared.i4_caches[layer_idx]
            append_quantized_kv_decode(
                k_cur,
                v_cur,
                metadata,
                kv_cache,
                kv_param,
                num_layers=1,
                layer_idx=0,
                apply_rope_to_k=False,
            )
            attn_out = decode_gqa_i4(q, kv_cache, kv_param, metadata, batch, q_heads, kv_heads)
            packed, scales = quantize_attention_output(attn_out.reshape(batch, cfg.hidden_size))
            o_in = dequant_grouped(packed, scales, GROUP_SIZE).reshape(batch, 1, cfg.hidden_size)
        else:
            raise ValueError(f"unsupported variant: {variant}")

        hidden = residual + layer.self_attn.o_proj(o_in)
        residual = hidden
        normed = layer.post_attention_layernorm(hidden)
        gate = layer.mlp.gate_proj(normed)
        up = layer.mlp.up_proj(normed)
        if variant == "fused_quarot":
            ffn_packed, ffn_scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(
                gate.reshape(batch, -1).contiguous(),
                up.reshape(batch, -1).contiguous(),
                GROUP_SIZE,
            )
            ffn_in = dequant_grouped(ffn_packed, ffn_scales, GROUP_SIZE).reshape(batch, 1, -1)
        elif variant == "quarot_unfused":
            ffn = F.silu(gate.float()) * up.float()
            ffn_packed, ffn_scales = quantize_grouped(
                hadamard(ffn.reshape(batch, -1, GROUP_SIZE)).reshape(batch, -1),
                GROUP_SIZE,
            )
            ffn_in = dequant_grouped(ffn_packed, ffn_scales, GROUP_SIZE).reshape(batch, 1, -1)
        else:
            ffn_in = F.silu(gate.float()).half() * up
        hidden = residual + layer.mlp.down_proj(ffn_in)

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits[:, -1]


def benchmark_paths(model, tokenizer, args, out_dir):
    prompt = load_prompts(args.prompts)[0]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    rows = []
    stat_rows = []
    correctness_rows = []
    for batch in parse_ints(args.batches):
        for context_len in parse_ints(args.context_lengths):
            prepared = prepare_inputs(model, tokenizer, prompt, batch, context_len, args.device)
            ref = run_manual_decode(model, prepared, "fp16_hf")
            outputs_by_variant = {}
            for variant in variants:
                out = ref if variant == "fp16_hf" else run_manual_decode(model, prepared, variant)
                outputs_by_variant[variant] = out
                metrics = logits_metrics(ref, out, topk=10)
                correctness_rows.append({
                    "batch": batch,
                    "context_len": context_len,
                    "variant": variant,
                    "reference": "fp16_hf",
                    **metrics,
                })
                repeat_vals = []
                for repeat in range(args.repeats):
                    ms = cuda_event_time_ms(lambda v=variant: run_manual_decode(model, prepared, v), args.warmup, args.iters)
                    repeat_vals.append(ms)
                    rows.append({
                        "variant": variant,
                        "batch": batch,
                        "context_len": context_len,
                        "repeat": repeat,
                        "decode_ms_per_token": ms,
                        "decode_tokens_per_sec": 1000.0 * batch / ms,
                    })
                stat_rows.append({
                    "variant": variant,
                    "batch": batch,
                    "context_len": context_len,
                    "metric": "decode_ms_per_token",
                    **summarize(repeat_vals),
                })
            if "quarot_unfused" in outputs_by_variant and "fused_quarot" in outputs_by_variant:
                metrics = logits_metrics(outputs_by_variant["quarot_unfused"], outputs_by_variant["fused_quarot"], topk=10)
                correctness_rows.append({
                    "batch": batch,
                    "context_len": context_len,
                    "variant": "fused_quarot",
                    "reference": "quarot_unfused",
                    **metrics,
                })
    write_csv(out_dir / "decode_latency_repeats.csv", rows)
    write_csv(out_dir / "decode_latency_summary.csv", stat_rows)
    write_csv(out_dir / "decode_correctness.csv", correctness_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--out-dir", default="llama31_decode_path_results")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--variants", default="fp16_manual,quarot_unfused,fused_quarot")
    parser.add_argument("--batches", default="1")
    parser.add_argument("--context-lengths", default="10,128")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_tokenizer_and_model(
        RuntimeConfig(
            model_id=args.model_id,
            dtype=args.dtype,
            device=args.device,
            attn_implementation=args.attn_implementation,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
    )
    benchmark_paths(model, tokenizer, args, out_dir)


if __name__ == "__main__":
    main()
