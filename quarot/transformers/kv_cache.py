from transformers.cache_utils import Cache
from typing import Optional, Tuple, Dict, Any
import math
import torch
from .. import _HIP
import functools
from fast_hadamard_transform import hadamard_transform
from quarot.functional.quantization import get_minq_maxq

@torch.jit.script
def asym_quantize_and_pack_i4(x: torch.Tensor):
    minq, maxq = get_minq_maxq(bits=4, sym=False)
    xmax = torch.amax(x, dim=-1, keepdim=True)
    xmin = torch.amin(x, dim=-1, keepdim=True)
    scale = ((xmax - xmin).clamp(min=1e-5) / maxq)
    zero = -xmin
    q = torch.clamp(torch.round((x + zero) / scale), 0, maxq)

    # pack int4
    q = q.to(dtype=torch.uint8)
    q = q[..., 0::2] | (q[..., 1::2] << 4)
    return q, scale, zero

def unpack_i4_and_asym_dequantize(q, scale, zero):
    #unpack int4
    assert q.dtype == torch.uint8
    q = torch.stack((q & 0x0f, (q >> 4) & 0x0f), dim=-1).view(*q.shape[:-1], q.shape[-1] * 2)
    return q * scale - zero

def matmul_had_HIP(X, dtype):
    n = X.shape[-1]
    # Flatten explicitly so every KV head/token is an independent FHT row.
    # The HIP backend otherwise mixes chunk row 1+ across leading axes.
    rows = X.to(dtype).contiguous().view(-1, n)
    # gfx1201 hadacore is exact for at most eight rows per dispatch.
    output = torch.cat([
        hadamard_transform(rows[start:start + 8], scale=1/math.sqrt(n))
        for start in range(0, rows.shape[0], 8)
    ], dim=0)
    return output.to(X.dtype).view(X.shape)


def init_kv_i4(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               seqlen_indptr, layer_idx):
    return _HIP.init_kv_i4(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        seqlen_indptr, layer_idx)


def append_kv_i4(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               layer_idx):
    return _HIP.append_kv_i4(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        layer_idx)


def fused_append_kv_i4(kv_data, kv_param,
                       kv_indptr, kv_indices,
                       last_page_offset, k, v,
                       num_layers, layer_idx, num_heads,
                       page_size, batch_size):
    """Decode-only K/V path matching this package's asymmetric FlashInfer cache."""
    return _HIP.fused_append_kv_i4(
        kv_data, kv_param, kv_indptr, kv_indices, last_page_offset, k, v,
        num_layers, layer_idx, num_heads, page_size, batch_size)

def batch_decode_i4(o, q, kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, layer_idx):
    return _HIP.batch_decode_i4(
        o, q, kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, layer_idx)

def batch_decode_i4_gqa(o, q, kv_data, kv_param,
               kv_indptr, kv_indices, last_page_offset, layer_idx):
    return _HIP.batch_decode_i4_gqa(
        o, q, kv_data, kv_param, kv_indptr, kv_indices,
        last_page_offset, layer_idx)


def init_kv_f16(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               seqlen_indptr, layer_idx):
    return _HIP.init_kv_f16(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        seqlen_indptr, layer_idx)


def append_kv_f16(kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, k,
               v, k_param, v_param,
               layer_idx):
    return _HIP.append_kv_f16(
        kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, k,
        v, k_param, v_param,
        layer_idx)

def batch_decode_f16(o, q, kv_data, kv_param,
               kv_indptr, kv_indices,
               last_page_offset, layer_idx):
    return _HIP.batch_decode_f16(
        o, q, kv_data, kv_param,
        kv_indptr, kv_indices,
        last_page_offset, layer_idx)

def batch_decode_f16_gqa(o, q, kv_data, kv_param,
               kv_indptr, kv_indices, last_page_offset, layer_idx):
    return _HIP.batch_decode_f16_gqa(
        o, q, kv_data, kv_param, kv_indptr, kv_indices,
        last_page_offset, layer_idx)


class _AttentionStub(object):
    def __init__(self, cache_page_size, device, n_layers, disable_quant, hadamard_dtype):
        self.cache_page_size = cache_page_size
        self.n_layers = n_layers
        self.disable_quant = disable_quant
        self.hadamard_dtype = hadamard_dtype

    def forward(self, q, num_kv_heads, attention_kwargs, layer_idx):
        batch_size, q_len, num_qo_heads, head_dim = q.shape
        q = q.view(batch_size * q_len, num_qo_heads, head_dim)
        if self.hadamard_dtype is not None:
            q = matmul_had_HIP(q, dtype=self.hadamard_dtype) 
        attn_output = torch.empty_like(q)
        if self.disable_quant:
            batch_decode = (batch_decode_f16_gqa if num_qo_heads != num_kv_heads
                            else batch_decode_f16)
        else:
            batch_decode = (batch_decode_i4_gqa if num_qo_heads != num_kv_heads
                            else batch_decode_i4)
        batch_decode(
            attn_output, q, 
            **attention_kwargs, layer_idx=layer_idx
        )
        attn_output = attn_output.view(batch_size, q_len, num_qo_heads, head_dim)
        return attn_output


class CacheTransaction:
    """Logical KV transaction; stale provisional slots are overwritten later."""

    def __init__(self, cache):
        if cache._transaction is not None:
            raise RuntimeError("a cache transaction is already active")
        self.cache = cache
        self.start_length = cache.length
        self.proposed_length = self.start_length
        self.closed = False
        cache._transaction = self

    @property
    def proposed_tokens(self):
        return self.proposed_length - self.start_length

    def commit(self, keep_tokens):
        if self.closed:
            raise RuntimeError("cache transaction is already closed")
        if not 0 <= keep_tokens <= self.proposed_tokens:
            raise ValueError(
                f"keep_tokens must be in [0, {self.proposed_tokens}]")
        self.cache.length = self.start_length + keep_tokens
        self.cache._transaction = None
        self.closed = True

    def rollback(self):
        self.commit(0)

    def __del__(self):
        if not self.closed:
            self.cache.length = self.start_length
            self.cache._transaction = None
            self.closed = True


class MultiLayerPagedKVCache4Bit(Cache):
    def __init__(
        self, batch_size, page_size, max_seq_len, 
        device, n_layers, num_heads, head_dim,
        num_kv_heads=None,
        native_gqa=True,
        fused_decode_append=True,
        disable_quant=False, hadamard_dtype=torch.float16 ):
        self.page_size = page_size
        self.n_layers = n_layers
        self.num_q_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.native_gqa = bool(native_gqa)
        self.fused_decode_append = bool(fused_decode_append)
        # Expanded MHA remains available as a correctness oracle/ablation.
        self.cache_heads = (self.num_kv_heads if self.native_gqa
                            else self.num_q_heads)
        # transformers.Cache exposes ``batch_size`` as a read-only property.
        # Store the value privately and expose it below so this cache keeps the
        # same public API without assigning to the base-class descriptor.
        self._batch_size = batch_size
        max_page_cnt = self.page_cnt_from_length(max_seq_len)
        self.disable_quant = disable_quant
        self.pages = torch.empty(
            (
                max_page_cnt * batch_size, 
                n_layers, 
                2, 
                self.cache_heads,
                page_size, 
                head_dim if disable_quant else head_dim // 2 
            ), 
            dtype=torch.float16 if disable_quant else torch.uint8, device=device)
        
        self.scales = torch.empty((max_page_cnt * batch_size, n_layers, 2, self.cache_heads, page_size, 2), dtype=torch.float16, device=device)
        self.page_size = page_size
        self.max_seq_len = max_seq_len
        self._needs_init = [True] * n_layers
        self.length = 0
        self.device = device
        self.hadamard_dtype = hadamard_dtype
        self._transaction = None
        self._stub = _AttentionStub(
            self.page_size, device, n_layers, 
            disable_quant=self.disable_quant, 
            hadamard_dtype=self.hadamard_dtype)

    def page_cnt_from_length(self, length):
        return (length + self.page_size - 1) // self.page_size

    @property
    def batch_size(self):
        return self._batch_size
    
    def _ensure_page_cnt_per_batch(self, expected_page_cnt_per_batch):
        expected_page_cnt = expected_page_cnt_per_batch * self.batch_size
        if expected_page_cnt <= self.pages.shape[0]:
            return
        raise NotImplementedError

    @property
    def seen_tokens(self):
        return self.length

    def begin(self):
        return CacheTransaction(self)
        
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        
        b_sz, added_length, num_heads, head_dim = key_states.shape
        if num_heads != self.num_kv_heads:
            raise ValueError("KV tensor head count does not match cache configuration")

        orig_key_states = key_states
        orig_value_states = value_states

        use_fused_append = (
            self.fused_decode_append and not self.disable_quant and
            cache_kwargs.get("attention_mask") is None
        )

        if use_fused_append and self.cache_heads != num_heads:
            repeats = self.cache_heads // num_heads
            key_states = key_states.repeat_interleave(repeats, dim=2)
            value_states = value_states.repeat_interleave(repeats, dim=2)
            num_heads = self.cache_heads

        if not use_fused_append and self.hadamard_dtype is not None:
            key_states = matmul_had_HIP(key_states, dtype=self.hadamard_dtype)

        if use_fused_append:
            # The HIP kernel applies the exact head-wise Hadamard, target-cache
            # asymmetric quantization, and writes directly to the selected page.
            pass
        elif self.disable_quant:
            k_scale = key_states.new_ones((b_sz, added_length, num_heads, 1))
            k_zero = key_states.new_zeros((b_sz, added_length, num_heads, 1))
            v_scale = value_states.new_ones((b_sz, added_length, num_heads, 1))
            v_zero = value_states.new_zeros((b_sz, added_length, num_heads, 1))
        else:
            key_states, k_scale, k_zero = asym_quantize_and_pack_i4(key_states)
            value_states, v_scale, v_zero = asym_quantize_and_pack_i4(value_states)

        if not use_fused_append and self.cache_heads != num_heads:
            repeats = self.cache_heads // num_heads
            key_states = key_states.repeat_interleave(repeats, dim=2)
            value_states = value_states.repeat_interleave(repeats, dim=2)
            k_scale = k_scale.repeat_interleave(repeats, dim=2)
            k_zero = k_zero.repeat_interleave(repeats, dim=2)
            v_scale = v_scale.repeat_interleave(repeats, dim=2)
            v_zero = v_zero.repeat_interleave(repeats, dim=2)
            num_heads = self.cache_heads

        if not use_fused_append:
            k_param = torch.cat([k_scale, k_zero], dim=-1).view(self.batch_size * added_length, num_heads, 2)
            v_param = torch.cat([v_scale, v_zero], dim=-1).view(self.batch_size * added_length, num_heads, 2)

        quantized_head_dim = self.pages.shape[-1]

        assert b_sz == self.batch_size
        if layer_idx == 0:
            current_length = self.length
            new_length = current_length + added_length
            self._ensure_page_cnt_per_batch(self.page_cnt_from_length(new_length))
            self.length = new_length
            if self._transaction is not None:
                self._transaction.proposed_length = new_length
        attention_mask = cache_kwargs.get("attention_mask")
        if self._needs_init[layer_idx]:
            self._needs_init[layer_idx] = False
            if use_fused_append:
                fused_append_kv_i4(
                    **self.get_cache_specs_for_flash_infer(None),
                    k=key_states.contiguous(), v=value_states.contiguous(),
                    num_layers=self.n_layers, layer_idx=layer_idx,
                    num_heads=num_heads, page_size=self.page_size,
                    batch_size=self.batch_size,
                )
                return orig_key_states, orig_value_states
            if attention_mask is not None:
                nonzero_indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten().view(-1, 1)
                key_states = key_states.view(self.batch_size * added_length, num_heads * quantized_head_dim)
                value_states = value_states.view(self.batch_size * added_length, num_heads * quantized_head_dim)
                key_states = torch.gather(key_states, 0, nonzero_indices.expand(-1, num_heads * quantized_head_dim))
                value_states = torch.gather(value_states, 0, nonzero_indices.expand(-1, num_heads * quantized_head_dim))

                k_param = k_param.view(self.batch_size * added_length, num_heads * 2)
                v_param = v_param.view(self.batch_size * added_length, num_heads * 2)
                k_param = torch.gather(k_param, 0, nonzero_indices.expand(-1, num_heads * 2))
                v_param = torch.gather(v_param, 0, nonzero_indices.expand(-1, num_heads * 2))

                seqlens_in_batch = torch.nn.functional.pad(torch.cumsum(attention_mask.sum(dim=-1, dtype=torch.int32), dim=0, dtype=torch.int32), (1, 0))
            else:
                seqlens_in_batch = torch.arange(self.batch_size + 1, device=self.device, dtype=torch.int) * added_length

            init_kv = init_kv_f16 if self.disable_quant else init_kv_i4
            init_kv(
                **self.get_cache_specs_for_flash_infer(attention_mask),
                k=key_states.view(-1, num_heads, quantized_head_dim), 
                v=value_states.view(-1, num_heads, quantized_head_dim), 
                k_param=k_param.view(-1, num_heads, 2), 
                v_param=v_param.view(-1, num_heads, 2),
                seqlen_indptr=seqlens_in_batch,
                layer_idx=layer_idx
            )
            return orig_key_states, orig_value_states
        else:
            specs = self.get_cache_specs_for_flash_infer(attention_mask)
            if use_fused_append:
                fused_append_kv_i4(
                    **specs, k=key_states.contiguous(),
                    v=value_states.contiguous(),
                    num_layers=self.n_layers, layer_idx=layer_idx, num_heads=num_heads,
                    page_size=self.page_size, batch_size=self.batch_size,
                )
            elif added_length == 1:
                append_kv = append_kv_f16 if self.disable_quant else append_kv_i4
                append_kv(
                    **specs,
                    k=key_states.view(self.batch_size, num_heads, quantized_head_dim),
                    v=value_states.view(self.batch_size, num_heads, quantized_head_dim),
                    k_param=k_param.view(-1, num_heads, 2),
                    v_param=v_param.view(-1, num_heads, 2),
                    layer_idx=layer_idx,
                )
            else:
                # FlashInfer's prefill append kernel writes the supplied suffix
                # at (final sequence length - suffix length). This provides a
                # single-launch provisional chunk append without reallocating
                # or copying the existing cache.
                seqlen_indptr = (
                    torch.arange(self.batch_size + 1, device=self.device,
                                 dtype=torch.int32) * added_length)
                init_kv = init_kv_f16 if self.disable_quant else init_kv_i4
                init_kv(
                    **specs,
                    k=key_states.view(-1, num_heads, quantized_head_dim),
                    v=value_states.view(-1, num_heads, quantized_head_dim),
                    k_param=k_param.view(-1, num_heads, 2),
                    v_param=v_param.view(-1, num_heads, 2),
                    seqlen_indptr=seqlen_indptr,
                    layer_idx=layer_idx,
                )
            attention_specs = (self.get_virtual_cache_specs(added_length)
                               if added_length > 1 else specs)
        return functools.partial(
            self._stub.forward,
            num_kv_heads=num_heads,
            attention_kwargs=(attention_specs if not self._needs_init[layer_idx]
                              else self.get_cache_specs_for_flash_infer(attention_mask)),
            layer_idx=layer_idx,
        )

    def get_virtual_cache_specs(self, chunk_length):
        """Build causal paged metadata for B*chunk independent decode rows."""
        if chunk_length < 1:
            raise ValueError("chunk_length must be positive")
        start = self.length - chunk_length
        indptr = [0]
        indices = []
        offsets = []
        for batch in range(self.batch_size):
            for token in range(chunk_length):
                seq_len = start + token + 1
                pages = self.page_cnt_from_length(seq_len)
                indices.extend(page * self.batch_size + batch
                               for page in range(pages))
                indptr.append(len(indices))
                offset = seq_len % self.page_size
                offsets.append(self.page_size if seq_len and offset == 0
                               else offset)
        return {
            "kv_data": self.pages,
            "kv_indptr": torch.tensor(indptr, device=self.device,
                                      dtype=torch.int32),
            "kv_indices": torch.tensor(indices, device=self.device,
                                       dtype=torch.int32),
            "last_page_offset": torch.tensor(offsets, device=self.device,
                                             dtype=torch.int32),
            "kv_param": self.scales,
        }
    
    def get_cache_specs_for_flash_infer(self, attention_mask):
        if attention_mask is not None:
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        else:
            seqlens_in_batch = torch.tensor([self.length], dtype=torch.int32, device=self.device).expand(self.batch_size)
        page_cnt = self.page_cnt_from_length(seqlens_in_batch)
        if (page_cnt[0] != page_cnt).any():
            raise NotImplementedError("Current implementation does not support the case where batches have different number of pages")
        page_cnt = page_cnt[0]
        page_ptr = seqlens_in_batch % self.page_size
        page_ptr = torch.where((seqlens_in_batch != 0) & (page_ptr == 0), self.page_size, page_ptr)
        return {
            f"kv_data": self.pages,
            f"kv_indptr": torch.arange(0, self.batch_size + 1, device=self.device, dtype=torch.int) * page_cnt, 
            f"kv_indices": (
                (torch.arange(page_cnt, device=self.device, dtype=torch.int) * self.batch_size).unsqueeze(0) + 
                torch.arange(self.batch_size, device=self.device, dtype=torch.int).unsqueeze(1)).view(-1), 
            f"last_page_offset": page_ptr, #torch.full((self.batch_size, ), page_ptr, device=self.device, dtype=torch.int),
            f"kv_param": self.scales, 
        }

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.length

    def get_mask_sizes(self, cache_position: torch.Tensor,
                       layer_idx: int) -> Tuple[int, int]:
        """Transformers >=4.53 causal-mask compatibility for the custom cache."""
        del layer_idx
        return self.length + cache_position.shape[0], 0

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        return None

    def to_legacy_cache(self):
        return self
