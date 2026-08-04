#!/usr/bin/env python3
"""Export the first Llama-3.1-8B packed QuaRot checkpoint.

RTN is intentionally exposed as the bring-up mode. GPTQ is rejected until its
calibration Hessians are wired; an RTN file is never mislabeled as GPTQ.
"""

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import torch

from .checkpoint import save_sharded_checkpoint
from .config import W4A4Config
from .packing import pack_weight_reference
from .rotation import rotate_llama31_model


def _store_linear(output: dict[str, torch.Tensor], prefix: str, weight: torch.Tensor) -> None:
    groups = weight.size(1) // 128
    qweight = torch.empty(weight.size(0) // 16, groups, 16, 64, dtype=torch.uint8)
    scales = torch.empty(weight.size(0), groups, dtype=torch.float16)
    for begin in range(0, weight.size(0), 256):
        end = min(begin + 256, weight.size(0))
        packed = pack_weight_reference(weight[begin:end], group_size=128)
        qweight[begin // 16 : end // 16].copy_(packed.qweight.cpu())
        scales[begin:end].copy_(packed.scales.cpu())
    output[f"{prefix}.qweight"] = qweight
    output[f"{prefix}.scales"] = scales


def _store_embedding(output: dict[str, torch.Tensor], prefix: str, weight: torch.Tensor) -> None:
    groups = weight.size(1) // 128
    qweight = torch.empty(weight.size(0), weight.size(1) // 2, dtype=torch.uint8)
    scales = torch.empty(weight.size(0), groups, dtype=torch.float16)
    for begin in range(0, weight.size(0), 256):
        end = min(begin + 256, weight.size(0))
        packed = pack_weight_reference(weight[begin:end], group_size=128)
        rows = packed.qweight.permute(0, 2, 1, 3).reshape(end - begin, weight.size(1) // 2)
        qweight[begin:end].copy_(rows.cpu())
        scales[begin:end].copy_(packed.scales.cpu())
    output[f"{prefix}.qweight"] = qweight
    output[f"{prefix}.scales"] = scales


@torch.inference_mode()
def build_rtn_state(model) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    _store_embedding(output, "model.embed_tokens", model.model.embed_tokens.weight.data)
    for index, layer in enumerate(model.model.layers):
        root = f"model.layers.{index}"
        qkv = torch.cat(
            (layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight), dim=0
        )
        _store_linear(output, f"{root}.self_attn.qkv_proj", qkv)
        _store_linear(output, f"{root}.self_attn.o_proj", layer.self_attn.o_proj.weight)
        gate_up = torch.cat((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight), dim=0)
        _store_linear(output, f"{root}.mlp.gate_up_proj", gate_up)
        _store_linear(output, f"{root}.mlp.down_proj", layer.mlp.down_proj.weight)
    _store_linear(output, "lm_head", model.lm_head.weight)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--quant-method", choices=("rtn", "gptq"), default="rtn")
    parser.add_argument("--model-revision", default="unknown")
    args = parser.parse_args()
    if args.quant_method == "gptq":
        raise NotImplementedError(
            "GPTQ export requires calibrated layer Hessians and is not allowed to silently fall back to RTN"
        )

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()
    if args.device != "cpu":
        model.to(args.device)
    rotate_llama31_model(model, seed=0)
    tensors = build_rtn_state(model)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    quant_config = replace(
        W4A4Config(),
        weight_quant_method="rtn",
        model_revision=args.model_revision,
        converter_commit=commit,
    )
    model_config = model.config.to_dict()
    model_config["architectures"] = ["QuaRotW4A4LlamaForCausalLM"]
    model_config["quantization_config_file"] = "quantization_config.json"
    save_sharded_checkpoint(
        tensors,
        args.output,
        model_config=model_config,
        quantization_config=quant_config,
    )
    size = sum(value.numel() * value.element_size() for value in tensors.values())
    print(json.dumps({"output": str(Path(args.output).resolve()), "tensor_bytes": size}, indent=2))


if __name__ == "__main__":
    main()
