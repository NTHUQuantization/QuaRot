#!/usr/bin/env python3
"""Measure decode without repeating or timing prompt prefill."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from .model import QuaRotW4A4LlamaForCausalLM


def _measure_w4(model, prompt, decode_tokens: int, repeats: int) -> dict:
    logits, cache = model.prefill(prompt, max_new_tokens=decode_tokens)
    prefix_length = prompt.size(1)
    first = logits[:, -1].argmax(-1, keepdim=True)
    measurements = []
    expected = None
    for repeat in range(repeats):
        cache.seq_len = prefix_length
        token = first.clone()
        generated = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(decode_tokens):
            generated.append(int(token.item()))
            current, cache = model.decode_one(token, cache)
            token = current.argmax(-1, keepdim=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if expected is None:
            expected = generated
        elif generated != expected:
            raise AssertionError("W4 repeated decode is not deterministic")
        measurements.append(
            {
                "repeat": repeat,
                "ms_per_token": elapsed * 1000 / decode_tokens,
                "tokens_per_s": decode_tokens / elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    return {"measurements": measurements, "generated_ids": expected}


def _measure_hf(model, prompt, decode_tokens: int, repeats: int) -> dict:
    with torch.inference_mode():
        initial = model(input_ids=prompt, use_cache=True, return_dict=True)
        prefix_cache = initial.past_key_values
        first = initial.logits[:, -1].argmax(-1, keepdim=True)
        measurements = []
        expected = None
        for repeat in range(repeats):
            past = prefix_cache
            token = first.clone()
            generated = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(decode_tokens):
                generated.append(int(token.item()))
                output = model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
                past = output.past_key_values
                token = output.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if expected is None:
                expected = generated
            elif generated != expected:
                raise AssertionError("HF repeated decode is not deterministic")
            measurements.append(
                {
                    "repeat": repeat,
                    "ms_per_token": elapsed * 1000 / decode_tokens,
                    "tokens_per_s": decode_tokens / elapsed,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
            )
    return {"measurements": measurements, "generated_ids": expected}


def _summarize(result: dict) -> None:
    result["median_tokens_per_s"] = statistics.median(
        item["tokens_per_s"] for item in result["measurements"]
    )
    result["median_ms_per_token"] = statistics.median(
        item["ms_per_token"] for item in result["measurements"]
    )
    result["peak_allocated_bytes"] = max(
        item["peak_allocated_bytes"] for item in result["measurements"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompt = (torch.arange(args.context_len, device="cuda") % 128000).long().unsqueeze(0)
    report = {
        "device": torch.cuda.get_device_name(0),
        "context_len": args.context_len,
        "decode_tokens": args.decode_tokens,
        "repeats": args.repeats,
    }
    w4 = QuaRotW4A4LlamaForCausalLM.from_quantized(args.checkpoint, device="cuda")
    report["w4a4kv4"] = _measure_w4(w4, prompt, args.decode_tokens, args.repeats)
    _summarize(report["w4a4kv4"])
    del w4
    gc.collect()
    torch.cuda.empty_cache()
    if args.hf_model:
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval().to("cuda")
        report["hf_fp16"] = _measure_hf(hf, prompt, args.decode_tokens, args.repeats)
        _summarize(report["hf_fp16"])
        report["decode_speedup"] = (
            report["w4a4kv4"]["median_tokens_per_s"] / report["hf_fp16"]["median_tokens_per_s"]
        )
        report["greedy_match"] = (
            report["w4a4kv4"]["generated_ids"] == report["hf_fp16"]["generated_ids"]
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
