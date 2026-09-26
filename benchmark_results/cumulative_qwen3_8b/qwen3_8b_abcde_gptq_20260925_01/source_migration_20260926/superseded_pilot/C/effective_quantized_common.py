"""Shared FIRST/basic HIP decoder runtime for Llama, Qwen2, and Qwen3.

This mirrors the final HIP model-family structure, but intentionally keeps the
runtime unfused: no fused QKV GEMM, no fused attention-output quantization, no
fused FFN, and no fused RMSNorm+quantization.  It is the common baseline layer
used by FIRST/basic HIP checkpoints and benchmarks.
"""
import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

import quarot
import quarot.transformers
from quarot.transformers.kv_cache import matmul_had_HIP


ALL_LAYERNORM_LAYERS.append(quarot.nn.RMSNorm)


def config_head_dim(config):
    return getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)


def _quantization_config(config):
    values = (
        float(getattr(config, "quarot_input_clip_ratio", 0.9)),
        float(getattr(config, "quarot_kv_clip_ratio", 1.0)),
        int(getattr(config, "quarot_kv_group_size", 128)),
    )
    input_clip, kv_clip, kv_group = values
    if not 0 < input_clip <= 1:
        raise ValueError("quarot_input_clip_ratio must be in (0, 1]")
    if not 0 < kv_clip <= 1:
        raise ValueError("quarot_kv_clip_ratio must be in (0, 1]")
    if kv_group < 1 or kv_group % 2:
        raise ValueError("quarot_kv_group_size must be a positive even integer")
    return values


def _asymmetric_group_i4_qdq(states, group_size, clip_ratio):
    """Paper KV fake quantization: asymmetric INT4 in 128-wide groups."""
    if states.shape[-1] % group_size != 0:
        raise ValueError(
            f"head_dim={states.shape[-1]} must be divisible by "
            f"KV group_size={group_size}")
    original_shape = states.shape
    grouped = states.float().reshape(*states.shape[:-1], -1, group_size)
    xmax = grouped.amax(dim=-1, keepdim=True) * clip_ratio
    xmin = grouped.amin(dim=-1, keepdim=True) * clip_ratio
    scale = (xmax - xmin).clamp_min(1e-5).div(15.0)
    zero = -xmin
    quantized = torch.clamp(torch.round((grouped + zero) / scale), 0, 15)
    return (quantized * scale - zero).to(states.dtype).reshape(original_shape)


class QuarotAttentionMixin:
    def _init_quarot_attention(self, quantized):
        self._quarot_quantized = quantized
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.head_dim = config_head_dim(self.config)
        self.hidden_size = self.config.hidden_size
        input_clip, kv_clip, kv_group = _quantization_config(self.config)
        self.kv_clip_ratio = kv_clip
        self.kv_group_size = kv_group
        self.quantizer = (
            quarot.nn.Quantizer(input_clip)
            if quantized else torch.nn.Identity())
        self.o_proj_hadamard = (
            quarot.nn.OnlineHadamard(self.num_heads)
            if quantized else torch.nn.Identity()
        )
        if quantized:
            self.q_proj = quarot.nn.Linear4bit.from_float(self.q_proj)
            self.k_proj = quarot.nn.Linear4bit.from_float(self.k_proj)
            self.v_proj = quarot.nn.Linear4bit.from_float(self.v_proj)
            self.o_proj = torch.nn.Sequential(
                quarot.nn.Quantizer(input_clip),
                quarot.nn.Linear4bit.from_float(self.o_proj),
            )

    def forward(self, hidden_states, position_embeddings=None,
                attention_mask=None, past_key_value=None,
                cache_position=None, output_attentions=False,
                position_ids=None, **kwargs):
        plural_cache = kwargs.pop("past_key_values", None)
        if past_key_value is None:
            past_key_value = plural_cache
        output_attentions = False
        if isinstance(hidden_states, quarot.PackedQuantizedTensor):
            bsz, q_len = hidden_states.quantized_x.shape[:2]
        else:
            bsz, q_len = hidden_states.shape[:2]

        hidden_states = self.quantizer(hidden_states)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim)

        if hasattr(self, "q_norm"):
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if position_embeddings is None:
            kv_seq_len = key_states.shape[1]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = position_embeddings

        query_states, key_states = self._quarot_apply_rotary(
            query_states, key_states, cos, sin, unsqueeze_dim=2)

        if past_key_value is None and getattr(self, "_synthetic_kv4", False):
            query_states = matmul_had_HIP(query_states, dtype=torch.float32)
            key_states = matmul_had_HIP(key_states, dtype=torch.float32)
            key_states = _asymmetric_group_i4_qdq(
                key_states, self.kv_group_size, self.kv_clip_ratio)
            value_states = _asymmetric_group_i4_qdq(
                value_states, self.kv_group_size, self.kv_clip_ratio)

        if past_key_value is None:
            attn_output = _flash_attention_forward(
                query_states, key_states, value_states, attention_mask,
                query_length=q_len, is_causal=True,
                dropout=0.0 if not self.training else self.attention_dropout,
                position_ids=position_ids,
                softmax_scale=self.scaling,
                sliding_window=getattr(self, "sliding_window", None),
                attn_implementation="flash_attention_2")
        else:
            cache_out = past_key_value.update(
                key_states, value_states, self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": cache_position,
                 "attention_mask": attention_mask})
            if isinstance(cache_out, tuple):
                cached_key, cached_value = cache_out
                attn_output = _flash_attention_forward(
                    query_states, cached_key, cached_value, attention_mask,
                    query_length=q_len, is_causal=True,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    position_ids=position_ids,
                    softmax_scale=self.scaling,
                    sliding_window=getattr(self, "sliding_window", None),
                    attn_implementation="flash_attention_2")
            else:
                attn_output = cache_out(query_states)

        attn_output = self.o_proj_hadamard(
            attn_output.transpose(-1, -2)).transpose(-1, -2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None if not output_attentions else None


class QuarotMLPMixin:
    def _init_quarot_mlp(self, config):
        input_clip, _, _ = _quantization_config(config)
        self.quantizer = quarot.nn.Quantizer(input_clip)
        up, gate, down = self.up_proj, self.gate_proj, self.down_proj
        self.ffn_rotation = getattr(
            config, "quarot_ffn_rotation", "cuda_tensor_product_v1")
        if self.ffn_rotation == "grouped_h256_v1":
            physical = quarot.functional.grouped_ffn_physical_width(
                self.intermediate_size)
            self.up_proj = quarot.nn.Linear4bit(
                up.in_features, physical, bias=up.bias is not None,
                dtype=up.weight.dtype)
            self.gate_proj = quarot.nn.Linear4bit(
                gate.in_features, physical, bias=gate.bias is not None,
                dtype=gate.weight.dtype)
            self.down_proj = quarot.nn.Linear4bit(
                physical, down.out_features, bias=down.bias is not None,
                dtype=down.weight.dtype)
            self.down_proj_hadamard = None
            # final HIP's fused SiLU-H256-quant kernel uses amax/7 here,
            # independently of the 0.9 clipping used by regular activations.
            self.down_proj_quantizer = quarot.nn.GroupQuantizer(
                256, 1.0)
        elif self.ffn_rotation == "cuda_tensor_product_v1":
            self.up_proj = quarot.nn.Linear4bit.from_float(up)
            self.gate_proj = quarot.nn.Linear4bit.from_float(gate)
            self.down_proj_hadamard = quarot.nn.OnlineHadamard(
                self.intermediate_size)
            if self.down_proj_hadamard.had_rem_dim is not None:
                self.down_proj_hadamard._non_persistent_buffers_set.discard("had_rem_dim")
            self.down_proj_quantizer = quarot.nn.Quantizer(input_clip)
            self.down_proj = quarot.nn.Linear4bit.from_float(down)
        else:
            raise ValueError(
                f"unsupported QuaRot FFN rotation {self.ffn_rotation!r}")

    def forward(self, x):
        x = self.quantizer(x)
        down_input = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        if self.ffn_rotation == "grouped_h256_v1":
            down_input = quarot.functional.matmul_grouped_h256(down_input)
        else:
            down_input = self.down_proj_hadamard(down_input)
        down_input = self.down_proj_quantizer(down_input)
        return self.down_proj(down_input)


class QuarotCausalLMMixin:
    def _init_quarot_model(self, attention_cls, mlp_cls=None):
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = attention_cls(self.config, layer_idx)
            if mlp_cls is not None:
                layer.mlp = mlp_cls(self.config)
                layer.input_layernorm = quarot.nn.RMSNorm(
                    self.config.hidden_size, eps=self.config.rms_norm_eps)
                layer.post_attention_layernorm = quarot.nn.RMSNorm(
                    self.config.hidden_size, eps=self.config.rms_norm_eps)
        if mlp_cls is not None:
            self.model.norm = quarot.nn.RMSNorm(
                self.config.hidden_size, eps=self.config.rms_norm_eps)
        self._expected_max_length = None

    def build_cache(self, batch_size, page_size, max_length):
        projection = self.model.layers[0].self_attn.v_proj
        if isinstance(projection, quarot.nn.Linear4bit):
            device, dtype = projection.weight.device, torch.float16
        else:
            device, dtype = projection.weight.device, projection.weight.dtype
        disable_quant = self.cache_dtype == "float16"
        return quarot.transformers.MultiLayerPagedKVCache4Bit(
            batch_size=batch_size, page_size=page_size,
            max_seq_len=max_length, device=device,
            n_layers=len(self.model.layers),
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=config_head_dim(self.config),
            disable_quant=disable_quant,
            hadamard_dtype=None if disable_quant else dtype)

    def _get_logits_processor(self, generation_config, *args, **kwargs):
        self._expected_max_length = generation_config.max_length
        return super()._get_logits_processor(generation_config, *args, **kwargs)

    def forward(self, input_ids=None, *args, past_key_values=None, **kwargs):
        use_cache = kwargs.get("use_cache", self.config.use_cache)
        if past_key_values is None and use_cache and input_ids is not None:
            max_length = self._expected_max_length or input_ids.shape[1]
            self._expected_max_length = None
            past_key_values = self.build_cache(
                input_ids.shape[0], max_length, max_length)
        return super().forward(
            input_ids, *args, past_key_values=past_key_values, **kwargs)
