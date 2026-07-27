from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch

try:
    from . import attention_fusion_hip
except ImportError:
    import attention_fusion_hip


HEAD_DIM = 128
ATTENTION_OUT_DIM = 4096
ATTENTION_OUT_GROUP = 256


@dataclass(frozen=True)
class PagedKVMetadata:
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_offset: torch.Tensor
    pages_per_batch: int
    total_pages: int
    page_size: int
    batch_size: int


def make_uniform_paged_kv_metadata(batch_size: int, seq_len: int, page_size: int, device) -> PagedKVMetadata:
    pages_per_batch = (seq_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_batch
    indptr = torch.arange(0, (batch_size + 1) * pages_per_batch, pages_per_batch, device=device, dtype=torch.int32)
    indices = torch.arange(total_pages, device=device, dtype=torch.int32)
    last = seq_len - (pages_per_batch - 1) * page_size
    last_page_offset = torch.full((batch_size,), last, device=device, dtype=torch.int32)
    return PagedKVMetadata(indptr, indices, last_page_offset, pages_per_batch, total_pages, page_size, batch_size)


def allocate_quantized_kv_cache(
    metadata: PagedKVMetadata,
    num_layers: int,
    num_heads: int,
    device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kv_data = torch.empty(
        (metadata.total_pages, num_layers, 2, num_heads, metadata.page_size, HEAD_DIM // 2),
        device=device,
        dtype=torch.uint8,
    )
    kv_param = torch.empty(
        (metadata.total_pages, num_layers, 2, num_heads, metadata.page_size, 2),
        device=device,
        dtype=torch.float16,
    )
    return kv_data, kv_param


def append_quantized_kv_decode(
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: PagedKVMetadata,
    kv_data: Optional[torch.Tensor] = None,
    kv_param: Optional[torch.Tensor] = None,
    *,
    num_layers: int = 1,
    layer_idx: int = 0,
    apply_rope_to_k: bool = False,
    rope_theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Append one decode token after head-wise Hadamard and dynamic INT4 quantization."""
    if key.dim() != 3 or key.size(-1) != HEAD_DIM:
        raise ValueError("key must have shape [batch, heads, 128]")
    if value.shape != key.shape:
        raise ValueError("value must have the same shape as key")
    batch_size, num_heads, _ = key.shape
    if batch_size != metadata.batch_size:
        raise ValueError("metadata batch size does not match key")
    if kv_data is None or kv_param is None:
        kv_data, kv_param = allocate_quantized_kv_cache(metadata, num_layers, num_heads, key.device)
    attention_fusion_hip.append_kv_had_quant_inplace(
        key.contiguous(),
        value.contiguous(),
        kv_data,
        kv_param,
        metadata.indptr,
        metadata.indices,
        metadata.last_page_offset,
        num_layers,
        layer_idx,
        num_heads,
        metadata.page_size,
        batch_size,
        apply_rope_to_k,
        rope_theta,
    )
    return kv_data, kv_param


def quantize_attention_output(attention_out: torch.Tensor, backend: str = "current") -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply block Hadamard over 256-wide groups of a 4096-wide attention output and pack INT4."""
    if attention_out.size(-1) != ATTENTION_OUT_DIM:
        raise ValueError("attention_out last dimension must be 4096")
    attention_out = attention_out.contiguous()
    if backend == "current":
        return attention_fusion_hip.output_had_quant(attention_out)
    if backend == "hadacore256":
        return attention_fusion_hip.output_had_quant_hadacore256(attention_out)
    if backend == "hadacore4096_experimental":
        return attention_fusion_hip.output_had_quant_hadacore4096_experimental(attention_out)
    raise ValueError(f"unknown K3 backend: {backend}")


def quantize_attention_output_hadacore256(attention_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_attention_output(attention_out, backend="hadacore256")


def quantize_attention_output_hadacore4096_experimental(
    attention_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_attention_output(attention_out, backend="hadacore4096_experimental")


def quantize_attention_output_inplace(
    attention_out: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    attention_fusion_hip.output_had_quant_inplace(attention_out.contiguous(), packed, scales)


def hadamard_reference(x: torch.Tensor) -> torch.Tensor:
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


def apply_rope_k_reference(k: torch.Tensor, pos: int, theta: float = 10000.0) -> torch.Tensor:
    x = k.float().reshape(*k.shape[:-1], -1, 2)
    pair_idx = torch.arange(x.size(-2), device=k.device, dtype=torch.float32)
    freq = (1.0 / theta) ** (2.0 * pair_idx / k.size(-1))
    angle = pos * freq
    c = torch.cos(angle)
    s = torch.sin(angle)
    y0 = x[..., 0] * c - x[..., 1] * s
    y1 = x[..., 0] * s + x[..., 1] * c
    return torch.stack((y0, y1), dim=-1).reshape_as(k).half()


def quantize_s4_reference(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    scale = (xf.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q = torch.round(xf / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).contiguous()
    return packed, torch.stack((scale, scale * 8.0), dim=-1).half()


def quantize_grouped_reference(x: torch.Tensor, group_size: int = ATTENTION_OUT_GROUP) -> Tuple[torch.Tensor, torch.Tensor]:
    grouped = x.float().reshape(-1, x.size(-1) // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1) / 7.0).clamp_min(1e-8)
    q = torch.round(grouped / scale[..., None]).clamp(-8, 7).to(torch.int16)
    u = (q + 8).to(torch.uint8)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).reshape(*x.shape[:-1], x.size(-1) // 2)
    scale = scale.reshape(*x.shape[:-1], x.size(-1) // group_size)
    return packed.contiguous(), scale.half()
