#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from .model import QuaRotW4A4LlamaForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    model = QuaRotW4A4LlamaForCausalLM.from_quantized(args.checkpoint, device="cuda")
    prompt = torch.arange(128, device="cuda").long().unsqueeze(0)
    logits, cache = model.prefill(prompt, max_new_tokens=2)
    token = logits[:, -1].argmax(-1, keepdim=True)
    model.decode_one(token, cache)
    token = (token + 1) % model.config.vocab_size
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        model.decode_one(token, cache)
    torch.cuda.synchronize()
    print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
