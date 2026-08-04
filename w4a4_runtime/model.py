from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint_tensors
from .linear import W4A4Linear, W4Embedding
from .ops import (
    append_kv_biased_native,
    append_kv_chunk_biased_native,
    cross_head_hadamard_native,
    qkv_rope_hadamard,
    quantize_kv_biased_native,
    rmsnorm_quantize,
    silu_hadamard_quantize,
    virtual_verify_metadata_native,
    w4a4_linear,
)


def unweighted_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(dtype)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    # x: [B, heads, tokens, 128], Llama uses split-half rotary pairs.
    half = x.size(-1) // 2
    frequency = 1.0 / (
        theta ** (torch.arange(0, x.size(-1), 2, device=x.device, dtype=torch.float32) / x.size(-1))
    )
    angles = positions.float().unsqueeze(-1) * frequency.unsqueeze(0)
    cos = angles.cos().to(x.dtype)[None, None, :, :]
    sin = angles.sin().to(x.dtype)[None, None, :, :]
    first, second = x[..., :half], x[..., half:]
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def _quantize_kv_biased(x: torch.Tensor, clip_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch oracle retained for CPU tests; production uses the fused pair kernel."""
    value = x.float()
    amax = value.abs().amax(dim=-1)
    scale = (amax * clip_ratio / 7.0).clamp_min(1.0e-6)
    scale = torch.where(amax == 0, torch.ones_like(scale), scale)
    quantized = torch.round(value / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int16)
    biased = (quantized + 8).to(torch.uint8)
    packed = biased[..., 0::2] | (biased[..., 1::2] << 4)
    params = torch.stack((scale, scale * 8.0), dim=-1).half()
    return packed.contiguous(), params.contiguous()


class W4A4LlamaLayer(torch.nn.Module):
    def __init__(self, config, *, device=None):
        super().__init__()
        self.qkv_proj = W4A4Linear(config.hidden_size, 6144, device=device)
        self.o_proj = W4A4Linear(config.hidden_size, config.hidden_size, device=device)
        self.gate_up_proj = W4A4Linear(config.hidden_size, 2 * config.intermediate_size, device=device)
        self.down_proj = W4A4Linear(config.intermediate_size, config.hidden_size, device=device)


@dataclass
class W4A4PagedCache:
    batch_size: int
    seq_len: int
    max_seq_len: int
    page_size: int
    allocation: object
    kv_data: list[torch.Tensor]
    kv_params: list[torch.Tensor]


class CacheTransaction:
    def __init__(self, cache: W4A4PagedCache, start_len: int, proposed_len: int):
        self.cache = cache
        self.start_len = start_len
        self.proposed_len = proposed_len
        self.closed = False
        cache.seq_len = start_len

    def commit(self, accepted_length: int) -> None:
        if self.closed:
            raise RuntimeError("cache transaction is already closed")
        proposed = self.proposed_len - self.start_len
        if accepted_length < 0 or accepted_length > proposed:
            raise ValueError(f"accepted_length must be in [0, {proposed}]")
        self.cache.seq_len = self.start_len + accepted_length
        self.closed = True

    def rollback(self) -> None:
        if self.closed:
            raise RuntimeError("cache transaction is already closed")
        self.cache.seq_len = self.start_len
        self.closed = True

    def __del__(self):
        if not self.closed:
            self.cache.seq_len = self.start_len
            self.closed = True


class QuaRotW4A4LlamaForCausalLM(torch.nn.Module):
    """Llama-3.1-8B runtime whose persistent model weights are all packed W4."""

    def __init__(self, config, quant_config, *, device=None):
        super().__init__()
        self.config = SimpleNamespace(**config) if isinstance(config, dict) else config
        self.quant_config = quant_config
        required = {
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        }
        for name, expected in required.items():
            if getattr(self.config, name) != expected:
                raise ValueError(f"{name} must be {expected}")
        self.embed_tokens = W4Embedding(self.config.vocab_size, self.config.hidden_size, device=device)
        self.layers = torch.nn.ModuleList(
            W4A4LlamaLayer(self.config, device=device) for _ in range(self.config.num_hidden_layers)
        )
        self.lm_head = W4A4Linear(self.config.hidden_size, self.config.vocab_size, device=device)

    @staticmethod
    def _load_linear(module: W4A4Linear, tensors: dict[str, torch.Tensor], prefix: str) -> None:
        module.qweight = tensors.pop(f"{prefix}.qweight")
        module.scales = tensors.pop(f"{prefix}.scales")

    @classmethod
    def from_quantized(cls, path: str | Path, device: str | torch.device = "cuda"):
        tensors, config, quant_config = load_checkpoint_tensors(path, device=device)
        # Meta construction avoids allocating a second ~4 GiB packed state before
        # the safetensors-backed buffers are installed.
        model = cls(config, quant_config, device="meta")
        model.embed_tokens.qweight = tensors.pop("model.embed_tokens.qweight")
        model.embed_tokens.scales = tensors.pop("model.embed_tokens.scales")
        for index, layer in enumerate(model.layers):
            root = f"model.layers.{index}"
            cls._load_linear(layer.qkv_proj, tensors, f"{root}.self_attn.qkv_proj")
            cls._load_linear(layer.o_proj, tensors, f"{root}.self_attn.o_proj")
            cls._load_linear(layer.gate_up_proj, tensors, f"{root}.mlp.gate_up_proj")
            cls._load_linear(layer.down_proj, tensors, f"{root}.mlp.down_proj")
        cls._load_linear(model.lm_head, tensors, "lm_head")
        return model.eval()

    @property
    def device(self):
        return self.embed_tokens.qweight.device

    def _new_cache(self, batch: int, prompt_len: int, max_new_tokens: int, page_size: int = 128):
        from attention_fusion import allocate_quantized_kv_cache, make_uniform_paged_kv_metadata

        max_seq_len = prompt_len + max_new_tokens
        allocation = make_uniform_paged_kv_metadata(batch, max_seq_len, page_size, self.device)
        data, params = [], []
        for _ in self.layers:
            layer_data, layer_params = allocate_quantized_kv_cache(allocation, 1, 8, self.device)
            data.append(layer_data)
            params.append(layer_params)
        return W4A4PagedCache(batch, prompt_len, max_seq_len, page_size, allocation, data, params)

    @staticmethod
    def _active_metadata(cache: W4A4PagedCache, seq_len: int):
        from llama31_quarot.hf_quarot_model import make_active_metadata

        return make_active_metadata(cache.allocation, seq_len, cache.kv_data[0].device)

    @staticmethod
    def _write_prompt_cache(cache, layer_idx, k, v):
        # k/v: [B, L, KV heads, 128], already in their final QuaRot basis.
        kp, ks, vp, vs = quantize_kv_biased_native(k, v, 0.95)
        pages_per_batch = cache.allocation.pages_per_batch
        for batch in range(cache.batch_size):
            for page_offset in range((k.size(1) + cache.page_size - 1) // cache.page_size):
                begin = page_offset * cache.page_size
                end = min(begin + cache.page_size, k.size(1))
                page = batch * pages_per_batch + page_offset
                cache.kv_data[layer_idx][page, 0, 0, :, : end - begin] = kp[batch, begin:end].transpose(0, 1)
                cache.kv_data[layer_idx][page, 0, 1, :, : end - begin] = vp[batch, begin:end].transpose(0, 1)
                cache.kv_params[layer_idx][page, 0, 0, :, : end - begin] = ks[batch, begin:end].transpose(0, 1)
                cache.kv_params[layer_idx][page, 0, 1, :, : end - begin] = vs[batch, begin:end].transpose(0, 1)

    @staticmethod
    def _append_cache(cache, layer_idx, k, v, position):
        # k/v: [B, KV heads, 128]
        page_ids = cache.allocation.indices[
            cache.allocation.indptr[:-1] + position // cache.page_size
        ]
        slot = position % cache.page_size
        append_kv_biased_native(
            k,
            v,
            cache.kv_data[layer_idx],
            cache.kv_params[layer_idx],
            page_ids,
            slot,
            cache.page_size,
            0.95,
        )

    @staticmethod
    def _append_cache_chunk(cache, layer_idx, k, v, start_position):
        """Write [B, K, H, D] provisional KV into capacity-owned pages."""
        append_kv_chunk_biased_native(
            k,
            v,
            cache.kv_data[layer_idx],
            cache.kv_params[layer_idx],
            cache.allocation.indptr,
            cache.allocation.indices,
            start_position,
            cache.page_size,
            0.95,
        )

    @staticmethod
    def _virtual_verify_metadata(cache, start_len: int, chunk_len: int):
        """Build B*K causal requests that alias prefix/provisional cache pages."""
        from attention_fusion import PagedKVMetadata

        device = cache.kv_data[0].device
        if device.type == "cuda":
            indptr, indices, last_page_offset = virtual_verify_metadata_native(
                cache.kv_data[0],
                cache.batch_size,
                start_len,
                chunk_len,
                cache.page_size,
                cache.allocation.pages_per_batch,
            )
        else:
            # CPU oracle for metadata unit tests; production verification uses
            # the single-launch HIP builder above.
            request_indices = []
            host_indptr = [0]
            host_last = []
            for batch in range(cache.batch_size):
                first_page = batch * cache.allocation.pages_per_batch
                for offset in range(chunk_len):
                    length = start_len + offset + 1
                    used_pages = (length + cache.page_size - 1) // cache.page_size
                    request_indices.append(torch.arange(first_page, first_page + used_pages, dtype=torch.int32))
                    host_indptr.append(host_indptr[-1] + used_pages)
                    host_last.append(length - (used_pages - 1) * cache.page_size)
            indptr = torch.tensor(host_indptr, dtype=torch.int32)
            indices = torch.cat(request_indices)
            last_page_offset = torch.tensor(host_last, dtype=torch.int32)
        return PagedKVMetadata(
            indptr=indptr,
            indices=indices,
            last_page_offset=last_page_offset,
            pages_per_batch=cache.allocation.pages_per_batch,
            total_pages=cache.allocation.total_pages,
            page_size=cache.page_size,
            batch_size=cache.batch_size * chunk_len,
        )

    def _project(self, layer, hidden, positions):
        qkv = self._rms_linear(layer.qkv_proj, hidden)
        return qkv_rope_hadamard(qkv, positions, self.config.rope_theta)

    def _finish_layer(self, layer, hidden, attention):
        # attention: [B, tokens, 32, 128], V already contains the per-head H128.
        attention = cross_head_hadamard_native(attention).reshape(*hidden.shape[:-1], 4096)
        hidden = hidden + layer.o_proj(attention)
        gate, up = self._rms_linear(layer.gate_up_proj, hidden).chunk(2, dim=-1)
        packed_ffn = silu_hadamard_quantize(gate, up)
        down = w4a4_linear(packed_ffn, layer.down_proj.packed_weight(), layer.down_proj.bias)
        down = down.reshape(*hidden.shape[:-1], layer.down_proj.out_features)
        return hidden + down

    def _rms_linear(self, module: W4A4Linear, hidden: torch.Tensor) -> torch.Tensor:
        packed = rmsnorm_quantize(hidden, self.config.rms_norm_eps)
        output = w4a4_linear(packed, module.packed_weight(), module.bias)
        return output.reshape(*hidden.shape[:-1], module.out_features)

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self._rms_linear(self.lm_head, hidden)

    @torch.inference_mode()
    def prefill(self, input_ids, max_new_tokens: int = 1, *, return_all_logits: bool = False):
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be [batch, tokens]")
        batch, tokens = input_ids.shape
        cache = self._new_cache(batch, tokens, max_new_tokens)
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(tokens, device=self.device)
        for index, layer in enumerate(self.layers):
            q, k, v = self._project(layer, hidden, positions)
            k_expanded = k.repeat_interleave(4, dim=1)
            v_expanded = v.repeat_interleave(4, dim=1)
            attention = F.scaled_dot_product_attention(q, k_expanded, v_expanded, is_causal=True)
            attention = attention.transpose(1, 2)
            hidden = self._finish_layer(layer, hidden, attention)
            self._write_prompt_cache(cache, index, k.transpose(1, 2), v.transpose(1, 2))
        logits = self._logits(hidden if return_all_logits else hidden[:, -1:])
        return logits, cache

    @torch.inference_mode()
    def decode_one(self, input_ids, cache: W4A4PagedCache):
        if input_ids.shape != (cache.batch_size, 1):
            raise ValueError(f"decode_one expects [{cache.batch_size}, 1]")
        if cache.seq_len >= cache.max_seq_len:
            raise ValueError("cache capacity exhausted")
        import flashinfer_test._HIP as flashinfer_hip

        position = cache.seq_len
        metadata = self._active_metadata(cache, position + 1)
        hidden = self.embed_tokens(input_ids)
        positions = torch.tensor([position], device=self.device)
        for index, layer in enumerate(self.layers):
            q, k, v = self._project(layer, hidden, positions)
            self._append_cache(cache, index, k[:, :, 0], v[:, :, 0], position)
            out = torch.empty_like(q[:, :, 0])
            flashinfer_hip.batch_decode_i4_gqa(
                out,
                q[:, :, 0].contiguous(),
                cache.kv_data[index],
                cache.kv_params[index],
                metadata.indptr,
                metadata.indices,
                metadata.last_page_offset,
                1,
                0,
                32,
                8,
                cache.page_size,
                cache.batch_size,
            )
            hidden = self._finish_layer(layer, hidden, out.unsqueeze(1))
        cache.seq_len += 1
        logits = self._logits(hidden)
        return logits[:, -1], cache

    @torch.inference_mode()
    def verify_chunk(self, input_ids, cache: W4A4PagedCache):
        """Verify a whole speculative chunk with one virtual GQA batch per layer."""
        if input_ids.dim() != 2 or input_ids.size(0) != cache.batch_size:
            raise ValueError("input_ids must be [batch, speculative_tokens]")
        if not 1 <= input_ids.size(1) <= 16:
            raise ValueError("speculative length must be in [1, 16]")
        start = cache.seq_len
        chunk = input_ids.size(1)
        if start + chunk > cache.max_seq_len:
            raise ValueError("cache capacity exhausted")
        import flashinfer_test._HIP as flashinfer_hip

        metadata = self._virtual_verify_metadata(cache, start, chunk)
        try:
            hidden = self.embed_tokens(input_ids)
            positions = torch.arange(start, start + chunk, device=self.device)
            for index, layer in enumerate(self.layers):
                q, k, v = self._project(layer, hidden, positions)
                self._append_cache_chunk(
                    cache,
                    index,
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    start,
                )
                virtual_q = q.transpose(1, 2).reshape(cache.batch_size * chunk, 32, 128).contiguous()
                out = torch.empty_like(virtual_q)
                flashinfer_hip.batch_decode_i4_gqa(
                    out,
                    virtual_q,
                    cache.kv_data[index],
                    cache.kv_params[index],
                    metadata.indptr,
                    metadata.indices,
                    metadata.last_page_offset,
                    1,
                    0,
                    32,
                    8,
                    cache.page_size,
                    cache.batch_size * chunk,
                )
                attention = out.view(cache.batch_size, chunk, 32, 128)
                hidden = self._finish_layer(layer, hidden, attention)
            logits = self._logits(hidden)
        except Exception:
            cache.seq_len = start
            raise
        transaction = CacheTransaction(cache, start, start + chunk)
        return logits, transaction
