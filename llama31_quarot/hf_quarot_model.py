#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

import flashinfer_test._HIP as flashinfer_hip
from single_decoder_layer_benchmark import (
    dequant_grouped,
    dequant_paged_i4,
    hadamard,
    make_paged_f16,
    make_paged_i4_unfused,
    quantize_grouped,
    quantize_s4,
)
from attention_fusion import append_quantized_kv_decode, make_uniform_paged_kv_metadata, quantize_attention_output
import ffn_fusion_hip
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


PAGE_SIZE = 128
GROUP_SIZE = 256


def cache_layer_kv(past_key_values, layer_idx):
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx], past_key_values.value_cache[layer_idx]
    return past_key_values[layer_idx]


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


def layer_project(layer, hidden, position_embeddings):
    attn = layer.self_attn
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    q = attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    k = attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    v = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1])
    return q.squeeze(2).contiguous(), k.squeeze(2).contiguous(), v.squeeze(2).contiguous()


@dataclass
class QuaRotCache:
    variant: str
    batch: int
    prompt_len: int
    seq_len: int
    max_seq_len: int
    metadata: object
    f16_caches: Optional[list]
    i4_caches: Optional[list]
    hf_past_key_values: Optional[object]


class QuaRotLlamaForCausalLM:
    """Formal token-by-token QuaRot wrapper around a HF LlamaForCausalLM model.

    The wrapper keeps HuggingFace weights/modules intact, uses HF for prompt
    prefill, then routes one-token decode through the GQA-aware K1/K2/K3/FFN
    paths validated in decode_paths.py.
    """

    def __init__(self, model, variant: str, page_size: int = PAGE_SIZE, group_size: int = GROUP_SIZE):
        if variant not in {"fp16_hf", "fp16_manual", "quarot_unfused", "fused_quarot"}:
            raise ValueError(f"unsupported variant: {variant}")
        self.model = model
        self.variant = variant
        self.page_size = page_size
        self.group_size = group_size
        self.config = model.config
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def prefill(self, input_ids, attention_mask=None, max_new_tokens: int = 1):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = self._make_cache(outputs.past_key_values, input_ids.size(0), input_ids.size(1), max_new_tokens)
        return outputs.logits, cache

    def _make_cache(self, hf_past_key_values, batch: int, prompt_len: int, max_new_tokens: int) -> QuaRotCache:
        if self.variant == "fp16_hf":
            return QuaRotCache(
                variant=self.variant,
                batch=batch,
                prompt_len=prompt_len,
                seq_len=prompt_len,
                max_seq_len=prompt_len + max_new_tokens,
                metadata=None,
                f16_caches=None,
                i4_caches=None,
                hf_past_key_values=hf_past_key_values,
            )

        max_seq_len = prompt_len + max_new_tokens
        metadata = make_uniform_paged_kv_metadata(batch, max_seq_len, self.page_size, self.device)
        f16_caches = [] if self.variant == "fp16_manual" else None
        i4_caches = [] if self.variant in {"quarot_unfused", "fused_quarot"} else None
        for layer_idx in range(self.config.num_hidden_layers):
            k, v = cache_layer_kv(hf_past_key_values, layer_idx)
            prompt_k = k.transpose(1, 2).contiguous()
            prompt_v = v.transpose(1, 2).contiguous()
            if f16_caches is not None:
                f16_caches.append(make_paged_f16(prompt_k, prompt_v, metadata=metadata))
            if i4_caches is not None:
                i4_caches.append(make_paged_i4_unfused(prompt_k, prompt_v, rope=False, metadata=metadata))
        return QuaRotCache(
            variant=self.variant,
            batch=batch,
            prompt_len=prompt_len,
            seq_len=prompt_len,
            max_seq_len=max_seq_len,
            metadata=metadata,
            f16_caches=f16_caches,
            i4_caches=i4_caches,
            hf_past_key_values=None,
        )

    @torch.inference_mode()
    def decode_one(self, input_ids, cache: QuaRotCache):
        if input_ids.dim() != 2 or input_ids.size(1) != 1:
            raise ValueError("decode_one expects input_ids with shape [batch, 1]")
        if cache.variant != self.variant:
            raise ValueError(f"cache variant {cache.variant} does not match wrapper variant {self.variant}")
        if cache.seq_len + 1 > cache.max_seq_len:
            raise ValueError(f"cache max_seq_len={cache.max_seq_len} exhausted")

        if self.variant == "fp16_hf":
            attention_mask = torch.ones((input_ids.size(0), cache.seq_len + 1), device=input_ids.device, dtype=torch.long)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache.hf_past_key_values,
                use_cache=True,
            )
            cache.hf_past_key_values = outputs.past_key_values
            cache.seq_len += 1
            return outputs.logits[:, -1], cache

        logits = self._decode_one_manual(input_ids, cache)
        cache.seq_len += 1
        return logits, cache

    def _decode_one_manual(self, input_ids, cache: QuaRotCache):
        cfg = self.config
        batch = input_ids.size(0)
        q_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads
        write_seq_len = cache.seq_len + 1

        hidden = self.model.model.embed_tokens(input_ids)
        cache_position = torch.full((batch, 1), cache.seq_len, device=input_ids.device, dtype=torch.long)
        position_embeddings = self.model.model.rotary_emb(hidden, cache_position)

        for layer_idx, layer in enumerate(self.model.model.layers):
            residual = hidden
            normed = layer.input_layernorm(hidden)
            q, k_cur, v_cur = layer_project(layer, normed, position_embeddings)

            if self.variant == "fp16_manual":
                kv_cache, kv_param = cache.f16_caches[layer_idx]
                append_f16_current(kv_cache, k_cur, v_cur, cache.metadata, write_seq_len)
                attn_out = decode_gqa_f16(q, kv_cache, kv_param, cache.metadata, batch, q_heads, kv_heads)
                o_in = attn_out.reshape(batch, 1, cfg.hidden_size)
            elif self.variant == "quarot_unfused":
                kv_cache, kv_param = cache.i4_caches[layer_idx]
                append_i4_unfused_current(kv_cache, kv_param, k_cur, v_cur, cache.metadata, write_seq_len)
                kv_dequant = dequant_paged_i4(kv_cache, kv_param)
                param_f16 = torch.zeros((*kv_param.shape[:-1], 2), device=q.device, dtype=torch.float16)
                attn_out = decode_gqa_f16(q, kv_dequant, param_f16, cache.metadata, batch, q_heads, kv_heads)
                packed, scales = quantize_grouped(
                    hadamard(attn_out.reshape(batch, cfg.hidden_size // self.group_size, self.group_size)).reshape(
                        batch, cfg.hidden_size
                    ),
                    self.group_size,
                )
                o_in = dequant_grouped(packed, scales, self.group_size).reshape(batch, 1, cfg.hidden_size)
            elif self.variant == "fused_quarot":
                kv_cache, kv_param = cache.i4_caches[layer_idx]
                append_quantized_kv_decode(
                    k_cur,
                    v_cur,
                    cache.metadata,
                    kv_cache,
                    kv_param,
                    num_layers=1,
                    layer_idx=0,
                    apply_rope_to_k=False,
                )
                attn_out = decode_gqa_i4(q, kv_cache, kv_param, cache.metadata, batch, q_heads, kv_heads)
                packed, scales = quantize_attention_output(attn_out.reshape(batch, cfg.hidden_size))
                o_in = dequant_grouped(packed, scales, self.group_size).reshape(batch, 1, cfg.hidden_size)
            else:
                raise ValueError(f"unsupported variant: {self.variant}")

            hidden = residual + layer.self_attn.o_proj(o_in)
            residual = hidden
            normed = layer.post_attention_layernorm(hidden)
            gate = layer.mlp.gate_proj(normed)
            up = layer.mlp.up_proj(normed)
            if self.variant == "fused_quarot":
                ffn_packed, ffn_scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(
                    gate.reshape(batch, -1).contiguous(),
                    up.reshape(batch, -1).contiguous(),
                    self.group_size,
                )
                ffn_in = dequant_grouped(ffn_packed, ffn_scales, self.group_size).reshape(batch, 1, -1)
            elif self.variant == "quarot_unfused":
                ffn = F.silu(gate.float()) * up.float()
                ffn_packed, ffn_scales = quantize_grouped(
                    hadamard(ffn.reshape(batch, -1, self.group_size)).reshape(batch, -1),
                    self.group_size,
                )
                ffn_in = dequant_grouped(ffn_packed, ffn_scales, self.group_size).reshape(batch, 1, -1)
            else:
                ffn_in = F.silu(gate.float()).half() * up
            hidden = residual + layer.mlp.down_proj(ffn_in)

        hidden = self.model.model.norm(hidden)
        logits = self.model.lm_head(hidden)
        return logits[:, -1]

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, max_new_tokens: int = 16, do_sample: bool = False):
        if do_sample:
            raise ValueError("sampling is not implemented; use greedy generation")
        logits, cache = self.prefill(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        generated = input_ids
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        for step in range(max_new_tokens):
            generated = torch.cat([generated, next_token], dim=1)
            if step == max_new_tokens - 1:
                break
            logits, cache = self.decode_one(next_token, cache)
            next_token = logits.argmax(dim=-1, keepdim=True)
        return generated


def wrap_model(model, mode: str) -> QuaRotLlamaForCausalLM:
    if mode == "fp16_hf":
        return QuaRotLlamaForCausalLM(model, "fp16_hf")
    return QuaRotLlamaForCausalLM(model, mode)
