#!/usr/bin/env python3
from contextlib import contextmanager
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
from attention_fusion import (
    PagedKVMetadata,
    append_quantized_kv_decode,
    make_uniform_paged_kv_metadata,
    quantize_attention_output,
)
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


@contextmanager
def profile_stage(stage_timer, name, layer_idx=None):
    if stage_timer is None:
        yield
        return
    with stage_timer.stage(name, layer_idx):
        yield


def layer_project(layer, hidden, position_embeddings, stage_timer=None, layer_idx=None):
    attn = layer.self_attn
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    with profile_stage(stage_timer, "q_proj", layer_idx):
        q = attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
    with profile_stage(stage_timer, "k_proj", layer_idx):
        k = attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
    with profile_stage(stage_timer, "v_proj", layer_idx):
        v = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    with profile_stage(stage_timer, "rope", layer_idx):
        q, k = apply_rotary_pos_emb(q, k, position_embeddings[0], position_embeddings[1])
    with profile_stage(stage_timer, "qkv_layout", layer_idx):
        q = q.squeeze(2).contiguous()
        k = k.squeeze(2).contiguous()
        v = v.squeeze(2).contiguous()
    return q, k, v


@dataclass(frozen=True)
class FusionConfig:
    k1: bool = False
    k2: bool = False
    k3: bool = False
    ffn: bool = False
    backend: str = "current"

    def __post_init__(self):
        if self.backend not in {"current", "hadacore256"}:
            raise ValueError(f"unsupported fusion backend: {self.backend}")

    @classmethod
    def from_name(cls, name: str):
        configs = {
            "unfused_INT4": cls(),
            "k1_fused": cls(k1=True),
            "k1_k2_fused": cls(k1=True, k2=True),
            "attention_fused_current": cls(k1=True, k2=True, k3=True),
            "full_fused_current": cls(k1=True, k2=True, k3=True, ffn=True),
            "full_fused_hadacore256": cls(
                k1=True,
                k2=True,
                k3=True,
                ffn=True,
                backend="hadacore256",
            ),
        }
        try:
            return configs[name]
        except KeyError as exc:
            raise ValueError(f"unknown fusion config: {name}") from exc


def make_active_metadata(allocation: PagedKVMetadata, seq_len: int, device) -> PagedKVMetadata:
    """Create a view exposing only the populated prefix of a capacity-sized cache."""
    used_pages = (seq_len + allocation.page_size - 1) // allocation.page_size
    chunks = [
        torch.arange(
            batch_idx * allocation.pages_per_batch,
            batch_idx * allocation.pages_per_batch + used_pages,
            device=device,
            dtype=torch.int32,
        )
        for batch_idx in range(allocation.batch_size)
    ]
    indices = torch.cat(chunks)
    indptr = torch.arange(
        0,
        (allocation.batch_size + 1) * used_pages,
        used_pages,
        device=device,
        dtype=torch.int32,
    )
    last = seq_len - (used_pages - 1) * allocation.page_size
    last_page_offset = torch.full(
        (allocation.batch_size,),
        last,
        device=device,
        dtype=torch.int32,
    )
    return PagedKVMetadata(
        indptr=indptr,
        indices=indices,
        last_page_offset=last_page_offset,
        pages_per_batch=used_pages,
        total_pages=allocation.total_pages,
        page_size=allocation.page_size,
        batch_size=allocation.batch_size,
    )


@dataclass
class QuaRotCache:
    variant: str
    batch: int
    prompt_len: int
    seq_len: int
    max_seq_len: int
    metadata: object
    metadata_views: Optional[dict]
    f16_caches: Optional[list]
    i4_caches: Optional[list]
    hf_past_key_values: Optional[object]


class QuaRotLlamaForCausalLM:
    """Formal token-by-token QuaRot wrapper around a HF LlamaForCausalLM model.

    The wrapper keeps HuggingFace weights/modules intact, uses HF for prompt
    prefill, then routes one-token decode through the GQA-aware K1/K2/K3/FFN
    paths validated in decode_paths.py.
    """

    def __init__(
        self,
        model,
        variant: str,
        page_size: int = PAGE_SIZE,
        group_size: int = GROUP_SIZE,
        fusion_backend: str = "current",
        fusion_config: Optional[FusionConfig] = None,
    ):
        if variant not in {"fp16_hf", "fp16_manual", "quarot_unfused", "fused_quarot"}:
            raise ValueError(f"unsupported variant: {variant}")
        if fusion_backend not in {"current", "hadacore256"}:
            raise ValueError(f"unsupported fusion backend: {fusion_backend}")
        self.model = model
        self.variant = variant
        self.page_size = page_size
        self.group_size = group_size
        self.fusion_backend = fusion_backend
        if fusion_config is None:
            fusion_config = FusionConfig(
                k1=variant == "fused_quarot",
                k2=variant == "fused_quarot",
                k3=variant == "fused_quarot",
                ffn=variant == "fused_quarot",
                backend=fusion_backend,
            )
        self.fusion_config = fusion_config
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
                metadata_views=None,
                f16_caches=None,
                i4_caches=None,
                hf_past_key_values=hf_past_key_values,
            )

        max_seq_len = prompt_len + max_new_tokens
        allocation_metadata = make_uniform_paged_kv_metadata(batch, max_seq_len, self.page_size, self.device)
        metadata_views = {
            seq_len: make_active_metadata(allocation_metadata, seq_len, self.device)
            for seq_len in range(prompt_len + 1, max_seq_len + 1)
        }
        f16_caches = [] if self.variant == "fp16_manual" else None
        i4_caches = [] if self.variant in {"quarot_unfused", "fused_quarot"} else None
        for layer_idx in range(self.config.num_hidden_layers):
            k, v = cache_layer_kv(hf_past_key_values, layer_idx)
            prompt_k = k.transpose(1, 2).contiguous()
            prompt_v = v.transpose(1, 2).contiguous()
            if f16_caches is not None:
                f16_caches.append(make_paged_f16(prompt_k, prompt_v, metadata=allocation_metadata))
            if i4_caches is not None:
                i4_caches.append(make_paged_i4_unfused(prompt_k, prompt_v, rope=False, metadata=allocation_metadata))
        return QuaRotCache(
            variant=self.variant,
            batch=batch,
            prompt_len=prompt_len,
            seq_len=prompt_len,
            max_seq_len=max_seq_len,
            metadata=allocation_metadata,
            metadata_views=metadata_views,
            f16_caches=f16_caches,
            i4_caches=i4_caches,
            hf_past_key_values=None,
        )

    @torch.inference_mode()
    def decode_one(
        self,
        input_ids,
        cache: QuaRotCache,
        *,
        advance_cache: bool = True,
        stage_timer=None,
        fusion_config: Optional[FusionConfig] = None,
    ):
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

        logits = self._decode_one_manual(input_ids, cache, stage_timer, fusion_config)
        if advance_cache:
            cache.seq_len += 1
        return logits, cache

    def _decode_one_manual(self, input_ids, cache: QuaRotCache, stage_timer=None, fusion_config=None):
        cfg = self.config
        fusion = fusion_config or self.fusion_config
        batch = input_ids.size(0)
        q_heads = cfg.num_attention_heads
        kv_heads = cfg.num_key_value_heads
        write_seq_len = cache.seq_len + 1
        metadata = cache.metadata_views[write_seq_len]

        with profile_stage(stage_timer, "embedding_rotary_setup"):
            hidden = self.model.model.embed_tokens(input_ids)
            cache_position = torch.full((batch, 1), cache.seq_len, device=input_ids.device, dtype=torch.long)
            position_embeddings = self.model.model.rotary_emb(hidden, cache_position)

        for layer_idx, layer in enumerate(self.model.model.layers):
            residual = hidden
            with profile_stage(stage_timer, "input_rmsnorm", layer_idx):
                normed = layer.input_layernorm(hidden)
            q, k_cur, v_cur = layer_project(layer, normed, position_embeddings, stage_timer, layer_idx)

            if self.variant == "fp16_manual":
                kv_cache, kv_param = cache.f16_caches[layer_idx]
                with profile_stage(stage_timer, "k1_fp16_append", layer_idx):
                    append_f16_current(kv_cache, k_cur, v_cur, metadata, write_seq_len)
                with profile_stage(stage_timer, "k2_fp16_decode", layer_idx):
                    attn_out = decode_gqa_f16(q, kv_cache, kv_param, metadata, batch, q_heads, kv_heads)
                o_in = attn_out.reshape(batch, 1, cfg.hidden_size)
            elif self.variant in {"quarot_unfused", "fused_quarot"}:
                kv_cache, kv_param = cache.i4_caches[layer_idx]
                with profile_stage(stage_timer, "k1_fused" if fusion.k1 else "k1_unfused", layer_idx):
                    if fusion.k1:
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
                    else:
                        append_i4_unfused_current(kv_cache, kv_param, k_cur, v_cur, metadata, write_seq_len)
                if fusion.k2:
                    with profile_stage(stage_timer, "k2_int4_decode", layer_idx):
                        attn_out = decode_gqa_i4(q, kv_cache, kv_param, metadata, batch, q_heads, kv_heads)
                else:
                    with profile_stage(stage_timer, "k2_cache_dequant", layer_idx):
                        kv_dequant = dequant_paged_i4(kv_cache, kv_param)
                        param_f16 = torch.zeros((*kv_param.shape[:-1], 2), device=q.device, dtype=torch.float16)
                    with profile_stage(stage_timer, "k2_fp16_decode", layer_idx):
                        attn_out = decode_gqa_f16(q, kv_dequant, param_f16, metadata, batch, q_heads, kv_heads)
                with profile_stage(stage_timer, "k3_fused" if fusion.k3 else "k3_unfused", layer_idx):
                    if fusion.k3:
                        packed, scales = quantize_attention_output(
                            attn_out.reshape(batch, cfg.hidden_size),
                            backend=fusion.backend,
                        )
                    else:
                        packed, scales = quantize_grouped(
                            hadamard(
                                attn_out.reshape(batch, cfg.hidden_size // self.group_size, self.group_size)
                            ).reshape(batch, cfg.hidden_size),
                            self.group_size,
                        )
                with profile_stage(stage_timer, "k3_dequant", layer_idx):
                    o_in = dequant_grouped(packed, scales, self.group_size).reshape(batch, 1, cfg.hidden_size)
            else:
                raise ValueError(f"unsupported variant: {self.variant}")

            with profile_stage(stage_timer, "o_proj_residual", layer_idx):
                hidden = residual + layer.self_attn.o_proj(o_in)
            residual = hidden
            with profile_stage(stage_timer, "post_attention_rmsnorm", layer_idx):
                normed = layer.post_attention_layernorm(hidden)
            with profile_stage(stage_timer, "gate_proj", layer_idx):
                gate = layer.mlp.gate_proj(normed)
            with profile_stage(stage_timer, "up_proj", layer_idx):
                up = layer.mlp.up_proj(normed)
            if self.variant in {"quarot_unfused", "fused_quarot"} and fusion.ffn:
                with profile_stage(stage_timer, "ffn_fused", layer_idx):
                    if fusion.backend == "hadacore256":
                        ffn_packed, ffn_scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant_hadacore256(
                            gate.reshape(batch, -1).contiguous(),
                            up.reshape(batch, -1).contiguous(),
                        )
                    else:
                        ffn_packed, ffn_scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(
                            gate.reshape(batch, -1).contiguous(),
                            up.reshape(batch, -1).contiguous(),
                            self.group_size,
                        )
                with profile_stage(stage_timer, "ffn_dequant", layer_idx):
                    ffn_in = dequant_grouped(ffn_packed, ffn_scales, self.group_size).reshape(batch, 1, -1)
            elif self.variant in {"quarot_unfused", "fused_quarot"}:
                with profile_stage(stage_timer, "ffn_unfused", layer_idx):
                    ffn = F.silu(gate.float()) * up.float()
                    ffn_packed, ffn_scales = quantize_grouped(
                        hadamard(ffn.reshape(batch, -1, self.group_size)).reshape(batch, -1),
                        self.group_size,
                    )
                with profile_stage(stage_timer, "ffn_dequant", layer_idx):
                    ffn_in = dequant_grouped(ffn_packed, ffn_scales, self.group_size).reshape(batch, 1, -1)
            else:
                with profile_stage(stage_timer, "ffn_fp16", layer_idx):
                    ffn_in = F.silu(gate.float()).half() * up
            with profile_stage(stage_timer, "down_proj_residual", layer_idx):
                hidden = residual + layer.mlp.down_proj(ffn_in)

        with profile_stage(stage_timer, "final_norm_lm_head"):
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


def wrap_model(
    model,
    mode: str,
    fusion_backend: str = "current",
    fusion_config: Optional[FusionConfig] = None,
) -> QuaRotLlamaForCausalLM:
    if mode == "fp16_hf":
        return QuaRotLlamaForCausalLM(model, "fp16_hf", fusion_backend="current")
    return QuaRotLlamaForCausalLM(
        model,
        mode,
        fusion_backend=fusion_backend,
        fusion_config=fusion_config,
    )
