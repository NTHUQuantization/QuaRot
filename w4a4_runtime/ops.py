from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from .hadamard import hadamard_transform
from .hadamard import paley_hadamard_28
from .packing import PackedActivation, PackedWeight, linear_reference, pack_activation_reference

try:
    from w4a4_kernels import w4a4_kernels_hip as _native
except ImportError:
    _native = None


def native_available() -> bool:
    return _native is not None


def quantize_activation(
    x: torch.Tensor,
    *,
    group_size: int = 128,
    clip_ratio: float = 0.9,
    require_native: bool = True,
) -> PackedActivation:
    flat = x.reshape(-1, x.size(-1)).contiguous()
    if _native is None or not flat.is_cuda:
        if require_native:
            reason = "is not built" if _native is None else "requires a HIP tensor"
            raise RuntimeError(f"w4a4_kernels_hip {reason}; production W4A4 has no FP16 fallback")
        return pack_activation_reference(flat, group_size=group_size, clip_ratio=clip_ratio)
    qdata, scales = _native.quantize_activation(flat, group_size, clip_ratio)
    return PackedActivation(qdata, scales, flat.size(0), flat.size(1), group_size)


def rmsnorm_quantize(
    x: torch.Tensor,
    eps: float,
    *,
    group_size: int = 128,
    clip_ratio: float = 0.9,
    require_native: bool = True,
) -> PackedActivation:
    """Unweighted RMSNorm followed by the production A4 packing contract.

    RMSNorm arithmetic intentionally remains FP32/FP16.  The persistent output
    is packed A4; there is no dequantized activation buffer in the linear path.
    """
    flat = x.reshape(-1, x.size(-1)).contiguous()
    if _native is not None and flat.is_cuda:
        qdata, scales = _native.rmsnorm_quantize(flat, eps, group_size, clip_ratio)
        return PackedActivation(qdata, scales, flat.size(0), flat.size(1), group_size)
    if require_native:
        reason = "is not built" if _native is None else "requires a HIP tensor"
        raise RuntimeError(f"w4a4_kernels_hip {reason}")
    value = flat.float()
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return pack_activation_reference(normalized.to(flat.dtype), group_size=group_size, clip_ratio=clip_ratio)


def hadamard_quantize(
    x: torch.Tensor,
    transform: str = "auto",
    *,
    group_size: int = 128,
    clip_ratio: float = 0.9,
    require_native: bool = True,
) -> PackedActivation:
    """Apply the supported normalized QuaRot transform and immediately pack A4."""
    if transform not in ("auto", "h32", "h128", "h4096", "h14336"):
        raise ValueError(f"unsupported Hadamard transform: {transform}")
    expected = {"h32": 32, "h128": 128, "h4096": 4096, "h14336": 14336}
    if transform != "auto" and x.size(-1) != expected[transform]:
        raise ValueError(f"{transform} requires last dimension {expected[transform]}")
    rotated = hadamard_transform(x)
    return quantize_activation(
        rotated.to(x.dtype),
        group_size=group_size,
        clip_ratio=clip_ratio,
        require_native=require_native,
    )


def silu_hadamard_quantize(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    group_size: int = 128,
    clip_ratio: float = 0.9,
    require_native: bool = True,
) -> PackedActivation:
    gate_flat = gate.reshape(-1, gate.size(-1)).contiguous()
    up_flat = up.reshape_as(gate_flat).contiguous()
    if gate_flat.shape != up_flat.shape or gate_flat.size(1) != 14336:
        raise ValueError("gate/up must have matching last dimension 14336")
    if _native is not None and gate_flat.is_cuda:
        h28 = paley_hadamard_28(device=gate_flat.device, dtype=torch.float32)
        qdata, scales = _native.silu_hadamard_quantize(
            gate_flat, up_flat, h28, group_size, clip_ratio
        )
        return PackedActivation(qdata, scales, gate_flat.size(0), gate_flat.size(1), group_size)
    if require_native:
        raise RuntimeError("fused SiLU-Hadamard quantization requires the HIP extension")
    transformed = hadamard_transform(F.silu(gate_flat.float()) * up_flat.float())
    return pack_activation_reference(transformed, group_size=group_size, clip_ratio=clip_ratio)


def w4a4_linear(
    x: PackedActivation,
    weight: PackedWeight,
    bias: torch.Tensor | None = None,
    *,
    require_native: bool = True,
) -> torch.Tensor:
    if x.cols != weight.in_features or x.group_size != weight.group_size:
        raise ValueError("activation and weight contracts do not match")
    if _native is None or not x.qdata.is_cuda:
        if require_native:
            reason = "is not built" if _native is None else "requires HIP operands"
            raise RuntimeError(f"w4a4_kernels_hip {reason}; production W4A4 has no dequant fallback")
        return linear_reference(x, weight, bias)
    bias_arg = bias if bias is not None else torch.empty(0, device=x.qdata.device, dtype=torch.float16)
    # gfx1201 end-to-end measurements show that two outputs per wave provide
    # the best occupancy / activation-reuse balance for M=1 decode.
    output_tile = int(os.environ.get("W4A4_OUTPUT_TILE", "2"))
    if output_tile not in (1, 2, 4, 8, 16):
        raise ValueError("W4A4_OUTPUT_TILE must be 1, 2, 4, 8, or 16")
    decode_waves = int(os.environ.get("W4A4_DECODE_WAVES", "1"))
    if decode_waves not in (1, 4):
        raise ValueError("W4A4_DECODE_WAVES must be 1 or 4")
    parallel_name = os.environ.get("W4A4_PARALLEL_KERNEL", "auto")
    if parallel_name not in ("auto", "wmma", "rows4", "rows8"):
        raise ValueError("W4A4_PARALLEL_KERNEL must be auto, wmma, rows4, or rows8")
    return _native.w4a4_linear(
        x.qdata,
        x.scales,
        weight.qweight,
        weight.scales,
        bias_arg,
        output_tile,
        decode_waves,
        {"rows4": 0, "wmma": 1, "auto": 2, "rows8": 3}[parallel_name],
    )


def embedding_lookup_w4(
    token_ids: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int = 128,
    require_native: bool = True,
) -> torch.Tensor:
    if _native is None or not token_ids.is_cuda:
        if require_native:
            reason = "is not built" if _native is None else "requires HIP tensors"
            raise RuntimeError(f"w4a4_kernels_hip {reason}")
        from .packing import unpack_int4

        selected_q = unpack_int4(qweight[token_ids.reshape(-1)]).float().reshape(-1, scales.size(1), group_size)
        selected_s = scales[token_ids.reshape(-1)].float().unsqueeze(-1)
        out = (selected_q * selected_s).reshape(*token_ids.shape, -1)
        return out.half()
    return _native.embedding_lookup_w4(token_ids.contiguous(), qweight, scales, group_size)


def qkv_rope_hadamard(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split fused QKV and apply RoPE+H128 to Q/K in one HIP launch."""
    if _native is None or not qkv.is_cuda:
        raise RuntimeError("fused QKV RoPE-Hadamard requires the HIP extension")
    if qkv.dim() != 3 or qkv.size(-1) != 6144:
        raise ValueError("qkv must be [batch, tokens, 6144]")
    return tuple(
        _native.qkv_rope_hadamard(
            qkv.contiguous(), positions.to(device=qkv.device, dtype=torch.int64).contiguous(), float(theta)
        )
    )


def cross_head_hadamard_native(attention: torch.Tensor) -> torch.Tensor:
    if _native is None or not attention.is_cuda:
        raise RuntimeError("fused cross-head Hadamard requires the HIP extension")
    if attention.shape[-2:] != (32, 128):
        raise ValueError("attention must end in [32, 128]")
    return _native.cross_head_hadamard(attention.contiguous())


def quantize_kv_biased_native(
    key: torch.Tensor,
    value: torch.Tensor,
    clip_ratio: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _native is None or not key.is_cuda:
        raise RuntimeError("fused KV4 packing requires the HIP extension")
    if key.shape != value.shape or key.size(-1) != 128:
        raise ValueError("key/value must have matching shape ending in 128")
    return tuple(
        _native.quantize_kv_biased(key.contiguous(), value.contiguous(), float(clip_ratio))
    )


def append_kv_biased_native(
    key: torch.Tensor,
    value: torch.Tensor,
    cache_data: torch.Tensor,
    cache_params: torch.Tensor,
    page_ids: torch.Tensor,
    slot: int,
    page_size: int,
    clip_ratio: float = 0.95,
) -> None:
    """Quantize one decode token and write KV4 directly into paged storage."""
    if _native is None or not key.is_cuda:
        raise RuntimeError("fused KV4 append requires the HIP extension")
    _native.append_kv_biased(
        key.contiguous(),
        value.contiguous(),
        cache_data,
        cache_params,
        page_ids.to(dtype=torch.int32).contiguous(),
        int(slot),
        int(page_size),
        float(clip_ratio),
    )


def append_kv_chunk_biased_native(
    key: torch.Tensor,
    value: torch.Tensor,
    cache_data: torch.Tensor,
    cache_params: torch.Tensor,
    allocation_indptr: torch.Tensor,
    allocation_indices: torch.Tensor,
    start_position: int,
    page_size: int,
    clip_ratio: float = 0.95,
) -> None:
    if _native is None or not key.is_cuda:
        raise RuntimeError("fused provisional KV4 append requires the HIP extension")
    _native.append_kv_chunk_biased(
        key.contiguous(),
        value.contiguous(),
        cache_data,
        cache_params,
        allocation_indptr.contiguous(),
        allocation_indices.contiguous(),
        int(start_position),
        int(page_size),
        float(clip_ratio),
    )


def virtual_verify_metadata_native(
    anchor: torch.Tensor,
    batch: int,
    start_len: int,
    chunk: int,
    page_size: int,
    pages_per_batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _native is None or not anchor.is_cuda:
        raise RuntimeError("fused virtual metadata requires the HIP extension")
    return tuple(
        _native.virtual_verify_metadata(
            anchor, int(batch), int(start_len), int(chunk), int(page_size), int(pages_per_batch)
        )
    )
