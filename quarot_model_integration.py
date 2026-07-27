from dataclasses import dataclass
import os
import sys
from typing import Tuple

import torch

ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "attention_fusion"))
sys.path.append(os.path.join(ROOT, "ffn_fusion"))

import flashinfer_test._HIP as flashinfer_hip
import ffn_fusion_hip
from attention_fusion import (
    allocate_quantized_kv_cache,
    append_quantized_kv_decode,
    make_uniform_paged_kv_metadata,
    quantize_attention_output,
)


@dataclass
class DecodeStepOutput:
    attention: torch.Tensor
    attention_packed: torch.Tensor
    attention_scales: torch.Tensor
    ffn_packed: torch.Tensor
    ffn_scales: torch.Tensor
    kv_data: torch.Tensor
    kv_param: torch.Tensor


class QuaRotFusedDecodeLayer:
    """Integration wrapper for the fused QuaRot decode kernels in this repo.

    The caller is still responsible for model projections. This layer consumes
    projected Q/K/V plus FFN gate/up activations and wires together:

    K1: append K/V with Hadamard + dynamic INT4 quantization
    K2: FlashInfer INT4 KV decode
    K3: attention output Hadamard + dynamic INT4 quantization
    FFN: SiLU(gate) * up + Hadamard + dynamic INT4 quantization
    """

    def __init__(
        self,
        *,
        batch_size: int,
        num_heads: int = 32,
        head_dim: int = 128,
        seq_len: int = 1,
        page_size: int = 128,
        num_layers: int = 1,
        layer_idx: int = 0,
        ffn_group_size: int = 256,
        k3_backend: str = "current",
        ffn_backend: str = "current",
        device: torch.device | str = "cuda",
    ):
        if head_dim != 128:
            raise ValueError("current HIP kernels support head_dim=128")
        if num_heads * head_dim != 4096:
            raise ValueError("K3 expects num_heads * head_dim == 4096")
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.page_size = page_size
        self.num_layers = num_layers
        self.layer_idx = layer_idx
        self.ffn_group_size = ffn_group_size
        self.k3_backend = k3_backend
        self.ffn_backend = ffn_backend
        self.device = torch.device(device)
        self.metadata = make_uniform_paged_kv_metadata(batch_size, seq_len, page_size, self.device)
        self.kv_data, self.kv_param = allocate_quantized_kv_cache(
            self.metadata, num_layers, num_heads, self.device
        )

    def reset_cache(self) -> None:
        self.kv_data, self.kv_param = allocate_quantized_kv_cache(
            self.metadata, self.num_layers, self.num_heads, self.device
        )

    def append_kv(self, key: torch.Tensor, value: torch.Tensor, *, apply_rope_to_k: bool = False) -> None:
        self.kv_data, self.kv_param = append_quantized_kv_decode(
            key,
            value,
            self.metadata,
            self.kv_data,
            self.kv_param,
            num_layers=self.num_layers,
            layer_idx=self.layer_idx,
            apply_rope_to_k=apply_rope_to_k,
        )

    def decode_attention(self, query: torch.Tensor) -> torch.Tensor:
        if query.shape != (self.batch_size, self.num_heads, self.head_dim):
            raise ValueError("query must have shape [batch, num_heads, 128]")
        out = torch.empty_like(query)
        flashinfer_hip.batch_decode_i4(
            out,
            query.contiguous(),
            self.kv_data,
            self.kv_param,
            self.metadata.indptr,
            self.metadata.indices,
            self.metadata.last_page_offset,
            self.num_layers,
            self.layer_idx,
            self.num_heads,
            self.page_size,
            self.batch_size,
        )
        return out

    def quantize_attention(self, attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return quantize_attention_output(
            attention.reshape(self.batch_size, self.num_heads * self.head_dim),
            backend=self.k3_backend,
        )

    def quantize_ffn(self, gate: torch.Tensor, up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ffn_backend == "current":
            packed, scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(
                gate.contiguous(), up.contiguous(), self.ffn_group_size
            )
        elif self.ffn_backend == "hadacore256":
            if self.ffn_group_size != 256:
                raise ValueError("hadacore256 FFN backend requires ffn_group_size=256")
            packed, scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant_hadacore256(
                gate.contiguous(), up.contiguous()
            )
        else:
            raise ValueError(f"unknown FFN backend: {self.ffn_backend}")
        return packed, scales

    def decode_step(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ffn_gate: torch.Tensor,
        ffn_up: torch.Tensor,
        *,
        apply_rope_to_k: bool = False,
    ) -> DecodeStepOutput:
        self.append_kv(key, value, apply_rope_to_k=apply_rope_to_k)
        attention = self.decode_attention(query)
        attention_packed, attention_scales = self.quantize_attention(attention)
        ffn_packed, ffn_scales = self.quantize_ffn(ffn_gate, ffn_up)
        return DecodeStepOutput(
            attention=attention,
            attention_packed=attention_packed,
            attention_scales=attention_scales,
            ffn_packed=ffn_packed,
            ffn_scales=ffn_scales,
            kv_data=self.kv_data,
            kv_param=self.kv_param,
        )
