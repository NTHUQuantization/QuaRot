from __future__ import annotations

import torch

from .hadamard import hadamard_transform, sylvester_hadamard


@torch.inference_mode()
def fuse_rmsnorm_scales(model) -> None:
    def scale_columns_(weight: torch.Tensor, scale: torch.Tensor, rows: int = 2048) -> None:
        for begin in range(0, weight.size(0), rows):
            end = min(begin + rows, weight.size(0))
            weight.data[begin:end].copy_(
                (weight.data[begin:end].float() * scale.float().unsqueeze(0)).to(weight.dtype)
            )

    for layer in model.model.layers:
        input_scale = layer.input_layernorm.weight.data.float()
        for linear in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            scale_columns_(linear.weight, input_scale)
        post_scale = layer.post_attention_layernorm.weight.data.float()
        for linear in (layer.mlp.gate_proj, layer.mlp.up_proj):
            scale_columns_(linear.weight, post_scale)
        layer.input_layernorm.weight.data.fill_(1)
        layer.post_attention_layernorm.weight.data.fill_(1)
    final_scale = model.model.norm.weight.data.float()
    scale_columns_(model.lm_head.weight, final_scale)
    model.model.norm.weight.data.fill_(1)


def _rotation_signs(seed: int, device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (4096,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
    return signs.to(device=device, dtype=torch.float32)


def _right_rotate(weight: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Compute W @ (H D) without materializing the dense 4096² rotation."""
    signs = signs.to(weight.device).unsqueeze(0)
    for begin in range(0, weight.size(0), 2048):
        end = min(begin + 2048, weight.size(0))
        rotated = hadamard_transform(weight[begin:end].float())
        weight[begin:end].copy_((rotated * signs).to(weight.dtype))
    return weight


def _left_rotate_transpose(weight: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Compute (H D)^T @ W = D H @ W using a structured transform."""
    rotated = hadamard_transform(weight.float().T).T
    return (rotated * signs.to(weight.device).unsqueeze(1)).to(weight.dtype)


def _rotate_v_heads(weight: torch.Tensor, head_dim: int = 128) -> torch.Tensor:
    heads = weight.size(0) // head_dim
    h = sylvester_hadamard(head_dim, device=weight.device, dtype=torch.float32)
    shaped = weight.float().reshape(heads, head_dim, weight.size(1))
    return torch.einsum("ij,hjk->hik", h, shaped).reshape_as(weight).to(weight.dtype)


def _right_full_hadamard(weight: torch.Tensor) -> torch.Tensor:
    # Applying H to every output-row is equivalent to transforming the last dimension.
    for begin in range(0, weight.size(0), 2048):
        end = min(begin + 2048, weight.size(0))
        weight[begin:end].copy_(hadamard_transform(weight[begin:end].float()).to(weight.dtype))
    return weight


@torch.inference_mode()
def rotate_llama31_model(model, *, seed: int = 0) -> torch.Tensor:
    config = model.config
    if config.hidden_size != 4096 or config.intermediate_size != 14336:
        raise ValueError("the first converter only supports Llama-3.1-8B shapes")
    if getattr(config, "head_dim", 128) != 128:
        raise ValueError("head_dim must be 128")
    fuse_rmsnorm_scales(model)
    signs = _rotation_signs(seed, model.model.embed_tokens.weight.device)

    model.model.embed_tokens.weight.data = _right_rotate(model.model.embed_tokens.weight.data, signs)
    model.lm_head.weight.data = _right_rotate(model.lm_head.weight.data, signs)
    for layer in model.model.layers:
        for linear in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            linear.weight.data = _right_rotate(linear.weight.data, signs)
        layer.self_attn.v_proj.weight.data = _rotate_v_heads(layer.self_attn.v_proj.weight.data)

        layer.self_attn.o_proj.weight.data = _left_rotate_transpose(layer.self_attn.o_proj.weight.data, signs)
        layer.self_attn.o_proj.weight.data = _right_full_hadamard(layer.self_attn.o_proj.weight.data)

        for linear in (layer.mlp.gate_proj, layer.mlp.up_proj):
            linear.weight.data = _right_rotate(linear.weight.data, signs)
        layer.mlp.down_proj.weight.data = _left_rotate_transpose(layer.mlp.down_proj.weight.data, signs)
        layer.mlp.down_proj.weight.data = _right_full_hadamard(layer.mlp.down_proj.weight.data)
    return signs.to(dtype=torch.int8, device="cpu")
