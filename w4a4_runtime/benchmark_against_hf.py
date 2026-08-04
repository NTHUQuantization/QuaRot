#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import QuaRotW4A4LlamaForCausalLM


TEXT = (
    "Efficient local language model inference requires careful control of memory traffic, "
    "quantization error, attention cache layout, and speculative verification. "
    "The system should report measured latency and quality instead of relying on estimates. "
)


def _peak() -> dict:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    encoded = tokenizer(TEXT * 16, return_tensors="pt", add_special_tokens=True).input_ids
    input_ids = encoded[:, : args.tokens].to("cuda")
    labels = input_ids[:, 1:].cpu()
    report = {"device": torch.cuda.get_device_name(0), "tokens": input_ids.size(1)}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    w4 = QuaRotW4A4LlamaForCausalLM.from_quantized(args.checkpoint, device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    w4_logits, w4_cache = w4.prefill(
        input_ids, max_new_tokens=args.decode_tokens, return_all_logits=True
    )
    torch.cuda.synchronize()
    w4_prefill_ms = (time.perf_counter() - started) * 1000
    w4_generated = []
    next_token = w4_logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.decode_tokens):
        w4_generated.append(int(next_token.item()))
        current, w4_cache = w4.decode_one(next_token, w4_cache)
        next_token = current.argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    w4_decode_ms = (time.perf_counter() - started) * 1000
    w4_cpu = w4_logits[:, :-1].float().cpu()
    report["w4"] = {
        "prefill_ms": w4_prefill_ms,
        "decode_total_ms": w4_decode_ms,
        "decode_ms_per_token": w4_decode_ms / args.decode_tokens,
        "generated_ids": w4_generated,
        "memory": _peak(),
    }
    del w4_logits, w4_cache, w4, current, next_token
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    hf = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = hf(input_ids=input_ids, use_cache=True, return_dict=True)
    torch.cuda.synchronize()
    hf_prefill_ms = (time.perf_counter() - started) * 1000
    hf_cpu = output.logits[:, :-1].float().cpu()
    past = output.past_key_values
    hf_generated = []
    next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.decode_tokens):
        hf_generated.append(int(next_token.item()))
        output = hf(input_ids=next_token, past_key_values=past, use_cache=True, return_dict=True)
        past = output.past_key_values
        next_token = output.logits[:, -1].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    hf_decode_ms = (time.perf_counter() - started) * 1000
    report["hf_fp16"] = {
        "prefill_ms": hf_prefill_ms,
        "decode_total_ms": hf_decode_ms,
        "decode_ms_per_token": hf_decode_ms / args.decode_tokens,
        "generated_ids": hf_generated,
        "memory": _peak(),
    }

    w4_loss = F.cross_entropy(w4_cpu.reshape(-1, w4_cpu.size(-1)), labels.reshape(-1))
    hf_loss = F.cross_entropy(hf_cpu.reshape(-1, hf_cpu.size(-1)), labels.reshape(-1))
    top1 = (w4_cpu.argmax(-1) == hf_cpu.argmax(-1)).float().mean()
    relative_l2 = (w4_cpu - hf_cpu).norm() / hf_cpu.norm().clamp_min(1e-12)
    report["quality"] = {
        "sample_tokens": int(labels.numel()),
        "w4_cross_entropy": float(w4_loss),
        "hf_cross_entropy": float(hf_loss),
        "w4_ppl": math.exp(float(w4_loss)),
        "hf_ppl": math.exp(float(hf_loss)),
        "ppl_ratio": math.exp(float(w4_loss - hf_loss)),
        "logits_relative_l2": float(relative_l2),
        "teacher_forced_top1_agreement": float(top1),
        "greedy_prefix_match": next(
            (index for index, pair in enumerate(zip(w4_generated, hf_generated)) if pair[0] != pair[1]),
            min(len(w4_generated), len(hf_generated)),
        ),
    }
    report["ratios"] = {
        "decode_speedup_w4_over_hf": report["hf_fp16"]["decode_ms_per_token"]
        / report["w4"]["decode_ms_per_token"],
        "prefill_speedup_w4_over_hf": hf_prefill_ms / w4_prefill_ms,
        "peak_allocated_ratio_w4_over_hf": report["w4"]["memory"]["peak_allocated_bytes"]
        / report["hf_fp16"]["memory"]["peak_allocated_bytes"],
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
