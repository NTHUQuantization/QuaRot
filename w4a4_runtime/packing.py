from dataclasses import dataclass

import torch


def _require_groupable(x: torch.Tensor, group_size: int) -> None:
    if x.size(-1) % group_size:
        raise ValueError(f"last dimension {x.size(-1)} must be divisible by {group_size}")
    if x.size(-1) % 2:
        raise ValueError("INT4 packing requires an even last dimension")


def _signed_to_nibble(x: torch.Tensor) -> torch.Tensor:
    return (x.to(torch.int16) & 0xF).to(torch.uint8)


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    if x.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("pack_int4 expects a signed integer tensor")
    if x.size(-1) % 2:
        raise ValueError("INT4 packing requires an even last dimension")
    if torch.any(x < -8) or torch.any(x > 7):
        raise ValueError("INT4 values must be in [-8, 7]")
    nibble = _signed_to_nibble(x)
    return (nibble[..., 0::2] | (nibble[..., 1::2] << 4)).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError("unpack_int4 expects uint8 storage")
    lo = (packed & 0xF).to(torch.int8)
    hi = ((packed >> 4) & 0xF).to(torch.int8)
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    out = torch.empty((*packed.shape[:-1], packed.size(-1) * 2), dtype=torch.int8, device=packed.device)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


@dataclass(frozen=True)
class PackedActivation:
    qdata: torch.Tensor
    scales: torch.Tensor
    rows: int
    cols: int
    group_size: int = 128

    def __post_init__(self) -> None:
        if self.qdata.dtype != torch.uint8 or self.scales.dtype != torch.float16:
            raise TypeError("activation storage must be uint8 with float16 scales")
        if self.qdata.shape != (self.rows, self.cols // 2):
            raise ValueError("invalid packed activation shape")
        if self.scales.shape != (self.rows, self.cols // self.group_size):
            raise ValueError("invalid activation scale shape")


@dataclass(frozen=True)
class PackedWeight:
    qweight: torch.Tensor
    scales: torch.Tensor
    out_features: int
    in_features: int
    group_size: int = 128

    def __post_init__(self) -> None:
        if self.out_features % 16:
            raise ValueError("out_features must be divisible by 16")
        if self.in_features % self.group_size:
            raise ValueError("in_features must be divisible by group_size")
        expected = (self.out_features // 16, self.in_features // self.group_size, 16, self.group_size // 2)
        if self.qweight.dtype != torch.uint8 or self.qweight.shape != expected:
            raise ValueError(f"qweight must be uint8 with shape {expected}")
        if self.scales.dtype != torch.float16 or self.scales.shape != (
            self.out_features,
            self.in_features // self.group_size,
        ):
            raise ValueError("invalid weight scale tensor")


def pack_activation_reference(
    x: torch.Tensor,
    *,
    group_size: int = 128,
    clip_ratio: float = 0.9,
) -> PackedActivation:
    if not x.is_floating_point():
        raise TypeError("activation must be floating point")
    if not 0.0 < clip_ratio <= 1.0:
        raise ValueError("clip_ratio must be in (0, 1]")
    _require_groupable(x, group_size)
    flat = x.reshape(-1, x.size(-1)).float()
    grouped = flat.reshape(flat.size(0), -1, group_size)
    amax = grouped.abs().amax(dim=-1)
    scales = (amax * clip_ratio / 7.0).clamp_min(1.0e-6)
    scales = torch.where(amax == 0, torch.ones_like(scales), scales)
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    packed = pack_int4(q.reshape(flat.size(0), -1))
    return PackedActivation(packed, scales.half(), flat.size(0), flat.size(1), group_size)


def pack_weight_reference(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
    clip_ratio: float = 1.0,
) -> PackedWeight:
    if weight.dim() != 2 or not weight.is_floating_point():
        raise TypeError("weight must be a floating [out_features, in_features] tensor")
    out_features, in_features = weight.shape
    if out_features % 16:
        raise ValueError("out_features must be divisible by 16")
    _require_groupable(weight, group_size)
    grouped = weight.float().reshape(out_features, -1, group_size)
    amax = grouped.abs().amax(dim=-1)
    scales = (amax * clip_ratio / 7.0).clamp_min(1.0e-6)
    scales = torch.where(amax == 0, torch.ones_like(scales), scales)
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    packed_rows = pack_int4(q.reshape(out_features, in_features))
    blocked = packed_rows.reshape(out_features // 16, 16, in_features // group_size, group_size // 2)
    blocked = blocked.permute(0, 2, 1, 3).contiguous()
    return PackedWeight(blocked, scales.half(), out_features, in_features, group_size)


def pack_embedding_reference(weight: torch.Tensor, *, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    packed = pack_weight_reference(weight, group_size=group_size)
    # Embedding lookup is row-oriented. Keep the same nibble convention without N16 blocking.
    rows = packed.qweight.permute(0, 2, 1, 3).reshape(weight.size(0), weight.size(1) // 2).contiguous()
    return rows, packed.scales


def dequantize_weight_reference(weight: PackedWeight) -> torch.Tensor:
    rows = weight.qweight.permute(0, 2, 1, 3).reshape(weight.out_features, weight.in_features // 2)
    q = unpack_int4(rows).float().reshape(weight.out_features, -1, weight.group_size)
    return (q * weight.scales.float().unsqueeze(-1)).reshape(weight.out_features, weight.in_features)


def linear_reference(x: PackedActivation, weight: PackedWeight, bias: torch.Tensor | None = None) -> torch.Tensor:
    if x.cols != weight.in_features or x.group_size != weight.group_size:
        raise ValueError("activation and weight contracts do not match")
    qx = unpack_int4(x.qdata).to(torch.int32).reshape(x.rows, -1, x.group_size)
    rows = weight.qweight.permute(0, 2, 1, 3).reshape(weight.out_features, weight.in_features // 2)
    qw = unpack_int4(rows).to(torch.int32).reshape(weight.out_features, -1, weight.group_size)
    partial = torch.einsum("mgk,ngk->mng", qx, qw).float()
    out = (partial * x.scales.float()[:, None, :] * weight.scales.float()[None, :, :]).sum(-1)
    if bias is not None:
        out = out + bias.float()
    return out.half()
