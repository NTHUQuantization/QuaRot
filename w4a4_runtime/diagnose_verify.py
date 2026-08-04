#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from .model import QuaRotW4A4LlamaForCausalLM


def _capture(model, storage):
    handles = []
    for layer_index, layer in enumerate(model.layers):
        for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            module = getattr(layer, name)

            def hook(_module, _inputs, output, layer_index=layer_index, name=name):
                storage.append((layer_index, name, output.detach().clone()))

            handles.append(module.register_forward_hook(hook))
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chunk", type=int, default=2)
    args = parser.parse_args()
    model = QuaRotW4A4LlamaForCausalLM.from_quantized(args.checkpoint, device="cuda")
    prompt = torch.arange(8, device="cuda").long().unsqueeze(0)
    candidates = torch.arange(17, 17 + args.chunk, device="cuda").long().unsqueeze(0)
    _, sequential_cache = model.prefill(prompt, max_new_tokens=args.chunk)
    sequential = []
    handles = _capture(model, sequential)
    for position in range(args.chunk):
        model.decode_one(candidates[:, position : position + 1], sequential_cache)
    for handle in handles:
        handle.remove()

    _, chunk_cache = model.prefill(prompt, max_new_tokens=args.chunk)
    chunk = []
    handles = _capture(model, chunk)
    _, transaction = model.verify_chunk(candidates, chunk_cache)
    transaction.rollback()
    for handle in handles:
        handle.remove()

    results = []
    projections_per_token = 32 * 4
    for index, (layer, name, chunk_value) in enumerate(chunk):
        for position in range(args.chunk):
            expected = sequential[position * projections_per_token + index][2]
            actual = chunk_value[:, position : position + 1]
            delta = (actual.float() - expected.float()).abs()
            results.append(
                {
                    "layer": layer,
                    "projection": name,
                    "position": position,
                    "max_abs": float(delta.max()),
                    "nonzero": int(torch.count_nonzero(delta)),
                }
            )
    print(json.dumps([item for item in results if item["nonzero"]][:20], indent=2))


if __name__ == "__main__":
    main()
