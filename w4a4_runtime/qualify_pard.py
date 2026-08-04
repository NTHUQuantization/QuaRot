#!/usr/bin/env python3
"""Fail-fast R9700 qualification required before packed-target PARD runs."""

from __future__ import annotations

import argparse
import json
import time

import torch

from .checkpoint import audit_checkpoint
from .model import QuaRotW4A4LlamaForCausalLM
from .ops import native_available


def _run_case(model, prompt_len: int, chunk_len: int) -> dict:
    device = model.device
    vocab = model.config.vocab_size
    prompt = (torch.arange(prompt_len, device=device) % (vocab - 1)).long().unsqueeze(0)
    candidates = ((torch.arange(chunk_len, device=device) + 17) % (vocab - 1)).long().unsqueeze(0)

    _, sequential_cache = model.prefill(prompt, max_new_tokens=chunk_len)
    sequential = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for offset in range(chunk_len):
        logits, sequential_cache = model.decode_one(candidates[:, offset : offset + 1], sequential_cache)
        sequential.append(logits)
    torch.cuda.synchronize()
    sequential_ms = (time.perf_counter() - started) * 1000
    sequential = torch.stack(sequential, dim=1)

    _, chunk_cache = model.prefill(prompt, max_new_tokens=chunk_len)
    torch.cuda.synchronize()
    started = time.perf_counter()
    chunk_logits, transaction = model.verify_chunk(candidates, chunk_cache)
    torch.cuda.synchronize()
    chunk_ms = (time.perf_counter() - started) * 1000
    position_diagnostics = []
    for position in range(chunk_len):
        delta = (chunk_logits[:, position].float() - sequential[:, position].float()).abs()
        position_diagnostics.append(
            {
                "position": position,
                "max_abs": float(delta.max()),
                "mean_abs": float(delta.mean()),
                "mismatched_at_0_05": int((delta > 0.05).sum()),
                "top1_equal": bool(
                    torch.equal(
                        chunk_logits[:, position].argmax(-1),
                        sequential[:, position].argmax(-1),
                    )
                ),
            }
        )
    try:
        torch.testing.assert_close(chunk_logits, sequential, rtol=1e-2, atol=5e-2)
    except AssertionError as exc:
        raise AssertionError(
            f"chunk verification mismatch: {json.dumps(position_diagnostics)}"
        ) from exc
    if not torch.equal(chunk_logits.argmax(-1), sequential.argmax(-1)):
        raise AssertionError("chunk and sequential greedy tokens differ")
    if chunk_cache.seq_len != prompt_len:
        raise AssertionError("provisional verification changed visible cache length")
    transaction.commit(chunk_len)
    if chunk_cache.seq_len != prompt_len + chunk_len:
        raise AssertionError("transaction commit produced the wrong cache length")

    _, rollback_cache = model.prefill(prompt, max_new_tokens=chunk_len)
    _, rollback = model.verify_chunk(candidates, rollback_cache)
    rollback.rollback()
    if rollback_cache.seq_len != prompt_len:
        raise AssertionError("rollback produced the wrong cache length")
    return {
        "prompt_len": prompt_len,
        "chunk_len": chunk_len,
        "sequential_ms": sequential_ms,
        "chunk_ms": chunk_ms,
        "speedup": sequential_ms / chunk_ms,
        "positions": position_diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunks", default="2,4,8,16")
    parser.add_argument("--skip-page-boundary", action="store_true")
    args = parser.parse_args()
    report = {"checkpoint": audit_checkpoint(args.checkpoint, strict_model=True)}
    if not native_available():
        raise RuntimeError("w4a4_kernels_hip is not built")
    if not torch.cuda.is_available():
        raise RuntimeError("R9700 qualification requires a visible ROCm GPU")
    device_name = torch.cuda.get_device_name(0)
    model = QuaRotW4A4LlamaForCausalLM.from_quantized(args.checkpoint, device=args.device)
    chunks = tuple(int(value) for value in args.chunks.split(",") if value)
    cases = [_run_case(model, 8, k) for k in chunks]
    if not args.skip_page_boundary:
        cases.append(_run_case(model, 127, 4))
    report.update(
        {
            "device": device_name,
            "cases": cases,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "status": "passed",
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
