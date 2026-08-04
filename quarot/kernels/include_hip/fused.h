#pragma once

#include <torch/extension.h>

void fused_append_kv_i4(
    torch::Tensor kv_data, torch::Tensor kv_param, torch::Tensor kv_indptr,
    torch::Tensor kv_indices, torch::Tensor last_page_offset, torch::Tensor key,
    torch::Tensor value, int64_t num_layers, int64_t layer_idx,
    int64_t num_heads, int64_t page_size, int64_t batch_size);

std::vector<torch::Tensor> fused_attention_hadamard_quant(
    torch::Tensor attention, int64_t num_heads);

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant(
    torch::Tensor gate, torch::Tensor up);

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_general(
    torch::Tensor gate, torch::Tensor up, torch::Tensor hadamard);

