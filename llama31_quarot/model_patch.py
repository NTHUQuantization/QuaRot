#!/usr/bin/env python3
"""Compatibility checks and integration notes for Llama-3.1 8B QuaRot fusion.

This module intentionally does not monkey-patch HuggingFace Llama attention.
The formal token-by-token wrapper in hf_quarot_model.py wires the validated
GQA-aware K1/K2/K3/FFN paths while keeping HF modules for prefill/projections.
"""

from dataclasses import dataclass

from .common import assert_llama31_kernel_compat, model_shape_summary


@dataclass
class IntegrationStatus:
    model_id: str
    compatible: bool
    reason: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int


def inspect_model_for_quarot(model, model_id: str) -> IntegrationStatus:
    shape = model_shape_summary(model)
    compatible, reason = assert_llama31_kernel_compat(model)
    return IntegrationStatus(model_id=model_id, compatible=compatible, reason=reason, **shape)


def require_fused_full_model_supported(model) -> None:
    compatible, reason = assert_llama31_kernel_compat(model)
    if not compatible:
        raise NotImplementedError(
            "Fused formal token-by-token Llama path is not compatible: "
            + reason
            + " Use fp16_hf for the baseline or inspect hf_quarot_model.py "
            "before enabling quarot_unfused/fused_quarot."
        )
