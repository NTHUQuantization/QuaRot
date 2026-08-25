"""Measure the isolated cost of PARD2-TD hidden-tap forward hooks.

This diagnostic deliberately excludes inverse-Hadamard restoration and the TD
projection.  It compares the same fused target call with no hooks, four no-op
hooks, and the reference-storing hooks used by ``SelectedHiddenCollector``.
Every timed call starts at the same logical cache length and is rolled back
afterwards, so all cases see identical tokens, positions, and KV history.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.benchmark_pard2 import preflight_gpu
from e2e.pard2 import DEFAULT_TARGET, DEFAULT_TOKENIZER, verify_target_checkpoint
from e2e.speculative import Pard2Spec, SelectedHiddenCollector, load_runtime


CASES = ("no_hooks", "noop_hooks", "store_ref_hooks")
DEFAULT_TEXT = (
    "Measure the overhead of hidden-state capture in speculative decoding. "
    "Keep the target input deterministic across every profiler case. "
)


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--target", default=str(DEFAULT_TARGET))
    result.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    result.add_argument("--iterations", type=positive_int, default=5)
    result.add_argument("--warmups", type=positive_int, default=2)
    result.add_argument("--output", required=True)
    result.add_argument("--context-length", type=positive_int, default=32)
    result.add_argument("--page-size", type=positive_int, default=128)
    result.add_argument("--max-preexisting-gib", type=float, default=1.0)
    return result


def summarize(samples):
    if not samples:
        raise ValueError("at least one latency sample is required")
    mean = statistics.mean(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": mean,
        "cv_percent": (
            100.0 * statistics.stdev(samples) / mean
            if len(samples) > 1 and mean else 0.0
        ),
    }


def interleaved_order(iteration):
    offset = iteration % len(CASES)
    return CASES[offset:] + CASES[:offset]


def _folded_hook_modules(target, taps):
    layers = target.model.layers
    indices = tuple(len(layers) + int(tap) for tap in taps)
    modules = []
    for position, index in enumerate(indices):
        if not 0 <= index < len(layers):
            raise ValueError(f"hidden tap {index} is outside {len(layers)} layers")
        modules.append(target.model.norm if position == 0 else layers[index])
    return tuple(modules)


@contextmanager
def installed_hooks(target, case, taps):
    handles = []
    collector = None
    try:
        if case == "noop_hooks":
            def no_op(_module, _inputs, _output):
                return None

            handles = [module.register_forward_hook(no_op)
                       for module in _folded_hook_modules(target, taps)]
        elif case == "store_ref_hooks":
            collector = SelectedHiddenCollector(
                target, taps, folded_basis=True)
            collector.reset()
        elif case != "no_hooks":
            raise ValueError(f"unknown profiler case: {case}")
        yield collector
    finally:
        if collector is not None:
            collector.reset()
            collector.close()
        for handle in handles:
            handle.remove()


def _fixed_tokens(tokenizer, needed, device):
    encoded = tokenizer(DEFAULT_TEXT, add_special_tokens=True,
                        return_tensors="pt").input_ids.flatten()
    if not encoded.numel():
        raise RuntimeError("tokenizer produced no input IDs")
    repeats = (needed + encoded.numel() - 1) // encoded.numel()
    return encoded.repeat(repeats)[:needed].view(1, -1).to(device)


def _target_call(runtime, ids, cache, positions):
    return runtime.target(
        input_ids=ids, past_key_values=cache, cache_position=positions,
        use_cache=True, attention_mask=None, return_dict=True,
        output_hidden_states=False)


@torch.inference_mode()
def _profile_call(runtime, ids, cache, positions, case, taps):
    start_length = cache.length
    transaction = cache.begin()
    try:
        with installed_hooks(runtime.target, case, taps) as collector:
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = _target_call(runtime, ids, cache, positions)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if collector is not None:
                missing = [index for index in collector.indices
                           if index not in collector.values]
                if missing:
                    raise RuntimeError(
                        f"store-ref hooks did not capture taps: {missing}")
            logits = output.logits.detach().clone()
    finally:
        transaction.rollback()
    if cache.length != start_length:
        raise RuntimeError("cache rollback did not restore the logical length")
    return elapsed_ms, logits


@torch.inference_mode()
def profile_shape(runtime, token_stream, q_len, iterations, warmups):
    context = token_stream[:, :-16]
    ids = token_stream[:, -q_len:].contiguous()
    cache = runtime._target_cache()
    context_positions = torch.arange(
        context.shape[1], device=token_stream.device)
    _target_call(runtime, context, cache, context_positions)
    torch.cuda.synchronize()
    base_length = cache.length
    positions = torch.arange(
        base_length, base_length + q_len, device=token_stream.device)
    taps = Pard2Spec().target_layers

    reference = None
    for iteration in range(warmups):
        for case in interleaved_order(iteration):
            _, logits = _profile_call(
                runtime, ids, cache, positions, case, taps)
            reference = logits if reference is None else reference

    samples = {case: [] for case in CASES}
    exact = {case: {"logits_exact": True, "top1_exact": True}
             for case in CASES}
    reference = None
    reference_top1 = None
    for iteration in range(iterations):
        for case in interleaved_order(iteration):
            elapsed_ms, logits = _profile_call(
                runtime, ids, cache, positions, case, taps)
            top1 = logits.argmax(dim=-1)
            if reference is None:
                reference = logits
                reference_top1 = top1
            exact[case]["logits_exact"] &= bool(torch.equal(logits, reference))
            exact[case]["top1_exact"] &= bool(torch.equal(top1, reference_top1))
            samples[case].append(elapsed_ms)

    if cache.length != base_length:
        raise RuntimeError("profile shape changed the base cache length")
    summaries = {case: {**summarize(values), **exact[case]}
                 for case, values in samples.items()}
    baseline = summaries["no_hooks"]["median_ms"]
    for case in CASES:
        median = summaries[case]["median_ms"]
        summaries[case]["relative_to_no_hooks"] = median / baseline
        summaries[case]["overhead_percent"] = 100.0 * (median / baseline - 1.0)
    return {
        "q_len": q_len,
        "context_length": base_length,
        "cases": summaries,
    }


@torch.inference_mode()
def main(argv=None):
    args = parser().parse_args(argv)
    if args.max_preexisting_gib < 0:
        raise ValueError("--max-preexisting-gib must be non-negative")
    gpu = preflight_gpu(int(args.max_preexisting_gib * (1 << 30)))
    verify_target_checkpoint(args.target)
    max_cache_len = args.context_length + 16
    runtime = load_runtime(
        mode="ar", target_checkpoint=args.target, draft_snapshot="",
        tokenizer_path=args.tokenizer, max_cache_len=max_cache_len,
        page_size=args.page_size, compile_mode="eager")
    token_stream = _fixed_tokens(
        runtime.tokenizer, args.context_length + 16, "cuda")
    results = [profile_shape(
        runtime, token_stream, q_len, args.iterations, args.warmups)
        for q_len in (1, 16)]
    payload = {
        "kind": "td_hidden_tap_hook_overhead",
        "contract": {
            "target": str(Path(args.target).resolve()),
            "tokenizer": str(Path(args.tokenizer).resolve()),
            "iterations": args.iterations,
            "warmups": args.warmups,
            "q_lengths": [1, 16],
            "cache_policy": "transaction rollback to fixed logical length",
            "timing": "interleaved wall clock with torch.cuda.synchronize",
            "store_ref_basis": "folded (final norm + three raw layer taps)",
        },
        "gpu_preflight": gpu,
        "results": results,
    }
    if not all(case[check]
               for shape in results
               for case in shape["cases"].values()
               for check in ("logits_exact", "top1_exact")):
        raise RuntimeError("a hidden-hook case changed target logits or top-1")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
