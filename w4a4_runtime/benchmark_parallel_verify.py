#!/usr/bin/env python3
"""A/B benchmark M=2..16 W4A4 target verification kernels in one process."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from .model import QuaRotW4A4LlamaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chunks", default="2,4,8,16")
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    chunks = [int(item) for item in args.chunks.split(",") if item]
    model = QuaRotW4A4LlamaForCausalLM.from_quantized(args.checkpoint, device="cuda")
    prompt = (torch.arange(args.context_len, device="cuda") % (model.config.vocab_size - 1)).long()[None]
    report = {
        "device": torch.cuda.get_device_name(0),
        "context_len": args.context_len,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "chunks": {},
    }
    for chunk in chunks:
        candidates = ((torch.arange(chunk, device="cuda") + 17) % (model.config.vocab_size - 1)).long()[None]
        outputs = {}
        timings = {}
        for implementation in ("rows4", "wmma", "auto"):
            os.environ["W4A4_PARALLEL_KERNEL"] = implementation
            _, cache = model.prefill(prompt, max_new_tokens=chunk)
            for _ in range(args.warmups):
                _, transaction = model.verify_chunk(candidates, cache)
                transaction.rollback()
            samples = []
            last_logits = None
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                started = time.perf_counter()
                last_logits, transaction = model.verify_chunk(candidates, cache)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - started) * 1000)
                transaction.rollback()
            outputs[implementation] = last_logits
            timings[implementation] = {
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
            }
        torch.testing.assert_close(outputs["wmma"], outputs["rows4"], rtol=1e-2, atol=5e-2)
        torch.testing.assert_close(outputs["auto"], outputs["rows4"], rtol=1e-2, atol=5e-2)
        timings["speedup_wmma_over_rows4"] = (
            timings["rows4"]["median_ms"] / timings["wmma"]["median_ms"]
        )
        timings["speedup_auto_over_rows4"] = (
            timings["rows4"]["median_ms"] / timings["auto"]["median_ms"]
        )
        report["chunks"][str(chunk)] = timings
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
