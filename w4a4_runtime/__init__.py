"""Packed W4A4 building blocks for the RDNA4 QuaRot runtime."""

from .config import W4A4Config
from .packing import (
    PackedActivation,
    PackedWeight,
    pack_activation_reference,
    pack_embedding_reference,
    pack_weight_reference,
    unpack_int4,
)
from .linear import W4A4Linear, W4Embedding
from .model import CacheTransaction, QuaRotW4A4LlamaForCausalLM, W4A4PagedCache
from .checkpoint import audit_checkpoint
from .ops import hadamard_quantize, quantize_activation, rmsnorm_quantize, w4a4_linear

__all__ = [
    "PackedActivation",
    "PackedWeight",
    "W4A4Config",
    "W4A4Linear",
    "W4Embedding",
    "CacheTransaction",
    "QuaRotW4A4LlamaForCausalLM",
    "W4A4PagedCache",
    "audit_checkpoint",
    "hadamard_quantize",
    "pack_activation_reference",
    "pack_embedding_reference",
    "pack_weight_reference",
    "quantize_activation",
    "rmsnorm_quantize",
    "unpack_int4",
    "w4a4_linear",
]
