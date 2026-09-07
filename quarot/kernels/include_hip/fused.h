#pragma once

#include <torch/extension.h>

const char* fused_fht_backend();

torch::Tensor fused_rope_append_kv_i4(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor cos, torch::Tensor sin, torch::Tensor kv_data,
    torch::Tensor kv_param, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor last_page_offset, int64_t num_layers, int64_t layer_idx,
    int64_t page_size);

std::vector<torch::Tensor> fused_attention_hadamard_quant(
    torch::Tensor attention, int64_t num_heads);

std::vector<torch::Tensor> fused_rmsnorm_quant_i4(
    torch::Tensor input, double eps, double clip_ratio);

std::vector<torch::Tensor> fused_attention_hadamard_quant_general(
    torch::Tensor attention, int64_t num_heads, torch::Tensor matrix);

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_grouped256(
    torch::Tensor gate, torch::Tensor up);

