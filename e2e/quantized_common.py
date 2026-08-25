"""Shared dense decoder runtime for rotated Llama, Qwen2, and Qwen3."""
import os
import torch
from transformers.modeling_flash_attention_utils import _flash_attention_forward
import quarot
import quarot.transformers

def config_head_dim(config):
    return getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)


class QuarotAttentionMixin:
    def _init_quarot_attention(self, quantized):
        self._quarot_quantized = quantized
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        self.quantizer = quarot.nn.Quantizer() if quantized else torch.nn.Identity()
        self.o_proj_hadamard = quarot.nn.OnlineHadamard(self.num_heads)
        if quantized:
            self.q_proj = quarot.nn.Linear4bit.from_float(self.q_proj)
            self.k_proj = quarot.nn.Linear4bit.from_float(self.k_proj)
            self.v_proj = quarot.nn.Linear4bit.from_float(self.v_proj)
            self.o_proj = torch.nn.Sequential(
                quarot.nn.Quantizer(), quarot.nn.Linear4bit.from_float(self.o_proj))
        self._fused_attention_output = quantized and self.hidden_size <= 8192

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, cache_position=None, **kwargs):
        if isinstance(hidden_states, quarot.PackedQuantizedTensor):
            bsz, q_len = hidden_states.quantized_x.shape[:2]
        else:
            bsz, q_len = hidden_states.shape[:2]
        hidden_states = self.quantizer(hidden_states)
        shape = (bsz, q_len, -1, self.head_dim)
        if self._quarot_quantized:
            query_states, key_states, value_states = quarot.nn.Linear4bit.fused_forward(
                hidden_states, self.q_proj, self.k_proj, self.v_proj)
            query_states = query_states.view(shape)
            key_states = key_states.view(shape)
            value_states = value_states.view(shape)
        else:
            query_states = self.q_proj(hidden_states).view(shape)
            key_states = self.k_proj(hidden_states).view(shape)
            value_states = self.v_proj(hidden_states).view(shape)
        if hasattr(self, "q_norm"):
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        cos, sin = position_embeddings
        use_fused_k1 = (
            past_key_value is not None and q_len == 1 and
            query_states.dtype == torch.float16 and
            cos.numel() == bsz * self.head_dim and
            hasattr(past_key_value, "can_fuse_k1") and
            past_key_value.can_fuse_k1(self.layer_idx, attention_mask))
        if use_fused_k1:
            query_states, cache_out = past_key_value.update_fused_k1(
                query_states, key_states, value_states, cos, sin,
                self.layer_idx)
            attn_output = cache_out(query_states)
        else:
            query_states, key_states = self._quarot_apply_rotary(
                query_states, key_states, cos, sin, unsqueeze_dim=2)
            if past_key_value is None:
                attn_output = _flash_attention_forward(
                    query_states, key_states, value_states, attention_mask,
                    query_length=q_len, is_causal=True,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    position_ids=kwargs.get("position_ids"),
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
                        position_ids=kwargs.get("position_ids"),
                        softmax_scale=self.scaling,
                        sliding_window=getattr(self, "sliding_window", None),
                        attn_implementation="flash_attention_2")
                else:
                    attn_output = cache_out(query_states)
        if self._fused_attention_output:
            if self.o_proj_hadamard.had_rem_dim is None:
                packed, scales = quarot._HIP.fused_attention_hadamard_quant(
                    attn_output.contiguous(), self.num_heads)
            else:
                packed, scales = quarot._HIP.fused_attention_hadamard_quant_general(
                    attn_output.contiguous(), self.num_heads,
                    self.o_proj_hadamard.had_rem_dim.contiguous())
            attn_output = self.o_proj[1](
                quarot.PackedQuantizedTensor(packed, scales))
        elif self._quarot_quantized:
            attn_output = self.o_proj_hadamard(
                attn_output.transpose(-1, -2)).transpose(-1, -2)
            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)
        else:
            # The FP16 baseline loads ordinary, unrotated Hugging Face
            # weights. Applying the online Hadamard here is only valid when
            # o_proj was transformed by the offline checkpoint converter.
            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = self.o_proj(attn_output)
        return attn_output, None

class QuarotMLPMixin:
    def _init_quarot_mlp(self):
        self.quantizer = quarot.nn.Quantizer()
        physical = quarot.functional.hadamard.grouped_ffn_physical_width(
            self.intermediate_size)
        up, gate, down = self.up_proj, self.gate_proj, self.down_proj
        self.ffn_physical_size = physical
        self.up_proj = quarot.nn.Linear4bit(
            up.in_features, physical, bias=up.bias is not None,
            dtype=up.weight.dtype)
        self.gate_proj = quarot.nn.Linear4bit(
            gate.in_features, physical, bias=gate.bias is not None,
            dtype=gate.weight.dtype)
        self.down_proj = quarot.nn.Linear4bit(
            physical, down.out_features, bias=down.bias is not None,
            dtype=down.weight.dtype)
        self._fused_ffn = True

    def _should_use_fused_ffn(self, x):
        return self._fused_ffn

    def forward(self, x):
        x = self.quantizer(x)
        gate, up = quarot.nn.Linear4bit.fused_forward(
            x, self.gate_proj, self.up_proj)
        packed, scales = (
            quarot._HIP.fused_ffn_silu_hadamard_quant(
                gate.contiguous(), up.contiguous()))
        return self.down_proj(quarot.PackedQuantizedTensor(packed, scales))

class QuarotCausalLMMixin:
    def _init_quarot_model(self, attention_cls, mlp_cls=None, norm_cls=None):
        for layer_idx, layer in enumerate(self.model.layers):
            layer.self_attn = attention_cls(self.config, layer_idx)
            if mlp_cls is not None:
                layer.mlp = mlp_cls(self.config)
                layer_norm_cls = norm_cls
                if (norm_cls is quarot.nn.FusedRMSNormQuant and
                        os.getenv("QUAROT_FUSED_NORM_QUANT", "1") == "0"):
                    layer_norm_cls = quarot.nn.RMSNorm
                layer.input_layernorm = layer_norm_cls(
                    self.config.hidden_size, eps=self.config.rms_norm_eps)
                layer.post_attention_layernorm = layer_norm_cls(
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
            batch_size=batch_size, page_size=page_size, max_seq_len=max_length,
            device=device, n_layers=len(self.model.layers),
            num_heads=self.config.num_attention_heads,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=config_head_dim(self.config), disable_quant=disable_quant,
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
