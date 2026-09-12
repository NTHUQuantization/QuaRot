#!/usr/bin/env python3
"""Benchmark a local INT4 checkpoint with the Fusion formal harness protocol."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

import quarot

try:
    from e2e.real_checkpoint import load_int4
    from e2e.model_registry import tokenizer_source
except ImportError:
    from real_checkpoint import load_int4
    from model_registry import tokenizer_source


DEFAULT_PROMPT = "The future of efficient large language model inference depends on"


def parse_ints(text):
    return [int(value) for value in text.replace(",", " ").split()]


def make_inputs(tokenizer, prompt, batch, context_len, device):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = encoded.input_ids[0]
    if ids.numel() < context_len:
        ids = torch.cat((ids, ids[-1:].repeat(context_len - ids.numel())))
    else:
        ids = ids[:context_len]
    input_ids = ids.unsqueeze(0).repeat(batch, 1).to(device)
    return input_ids, torch.ones_like(input_ids)


def summarize(values):
    ordered = sorted(float(value) for value in values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - position) + ordered[high] * (position - low)

    return {
        "samples_ms": ordered,
        "mean_ms": statistics.mean(ordered),
        "median_ms": statistics.median(ordered),
        "stddev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "min_ms": min(ordered),
        "max_ms": max(ordered),
    }


def cuda_event_time_ms(function, warmup_steps, timed_steps):
    for _ in range(warmup_steps):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timed_steps):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def remap_legacy_down_projections(model, checkpoint):
    """Load direct down_proj tensors into legacy Sequential down_proj[2]."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    requests = {}
    for layer_index, layer in enumerate(model.model.layers):
        if not isinstance(layer.mlp.down_proj, torch.nn.Sequential):
            continue
        projection = layer.mlp.down_proj[2]
        for suffix, destination in (
            ("weight", projection.weight),
            ("weight_scales", projection.weight_scales),
        ):
            key = f"model.layers.{layer_index}.mlp.down_proj.{suffix}"
            requests.setdefault(weight_map[key], []).append((key, destination))
    copied = 0
    for shard, tensors in requests.items():
        with safe_open(checkpoint / shard, framework="pt", device="cpu") as handle:
            for key, destination in tensors:
                source = handle.get_tensor(key)
                if source.shape != destination.shape:
                    raise RuntimeError(
                        f"legacy remap shape mismatch for {key}: "
                        f"{source.shape} != {destination.shape}")
                destination.copy_(source)
                copied += 1
    return copied


@torch.inference_mode()
def benchmark_shape(model, tokenizer, args, batch, context_len, decode_steps):
    input_ids, attention_mask = make_inputs(
        tokenizer, args.prompt, batch, context_len, torch.device("cuda"))
    next_ids = input_ids[:, -1:]
    capacity = args.warmup + decode_steps + 1
    prefill_samples = []
    decode_samples = []

    def fresh_prefill():
        model._expected_max_length = context_len + capacity
        return model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )

    for _ in range(args.repeats):
        torch.cuda.empty_cache()
        if args.measure_prefill:
            prefill_samples.append(cuda_event_time_ms(
                fresh_prefill, args.warmup, 1))

        seed = fresh_prefill()
        cache = seed.past_key_values
        torch.cuda.synchronize()

        def decode_one():
            return model(
                next_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )

        if args.cuda_graph:
            for _ in range(args.warmup):
                decode_one()
            cache.enable_cuda_graph_decode()
            cache_position = torch.tensor(
                [cache.length], device="cuda", dtype=torch.long)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                model(
                    next_ids,
                    past_key_values=cache,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache.advance_cuda_graph_decode()
                cache_position.add_(1)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(decode_steps):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            elapsed_ms = cuda_event_time_ms(
                decode_one, args.warmup, decode_steps)
        decode_samples.append(elapsed_ms / decode_steps)

    result = {
        "batch": batch,
        "context_len": context_len,
        "decode_steps": decode_steps,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "decode": summarize(decode_samples),
    }
    result["decode"]["mean_batch_tokens_per_second"] = (
        batch * 1000.0 / result["decode"]["mean_ms"])
    result["decode"]["mean_ms_per_output_token"] = (
        result["decode"]["mean_ms"] / batch)
    if prefill_samples:
        result["prefill"] = summarize(prefill_samples)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batches", default="1")
    parser.add_argument("--context-lengths", default="10,128,1024,2048,4096")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--kv-cache-dtype", choices=("int4", "float16"), default="int4",
        help="KV-cache storage used by the packed model (default: int4).")
    parser.add_argument(
        "--decode-steps", default="3",
        help="Comma-separated timed decode horizons; each result is ms/step.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measure-prefill", action="store_true")
    parser.add_argument(
        "--cuda-graph", action="store_true",
        help="Capture and replay a state-advancing decode step.")
    parser.add_argument(
        "--remap-legacy-down-proj", action="store_true",
        help="Map direct checkpoint down_proj tensors into legacy down_proj[2].")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmark_results/performance/fusion/fusion_harness_local_results.json"))
    args = parser.parse_args()
    decode_steps = parse_ints(args.decode_steps)
    if args.warmup < 0 or args.repeats <= 0 or not decode_steps or min(decode_steps) <= 0:
        raise ValueError(
            "warmup must be nonnegative; repeats/decode steps must be positive")

    tokenizer_model = tokenizer_source(
        args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model, local_files_only=args.local_files_only)
    model = load_int4(args.model)
    model.cache_dtype = args.kv_cache_dtype
    model = model.eval()
    if args.remap_legacy_down_proj:
        copied = remap_legacy_down_projections(model, args.model)
        expected = 2 * len(model.model.layers)
        if copied != expected:
            raise RuntimeError(f"legacy remap copied {copied} tensors; expected {expected}")
    model = model.cuda()
    for module in model.modules():
        if isinstance(module, quarot.nn.Linear4bit):
            module._prepack_weight()
    torch.cuda.synchronize()

    results = {
        "model": args.model,
        "tokenizer": tokenizer_model,
        "weight_dtype": "int4",
        "cache_dtype": model.cache_dtype,
        "timing": "CUDA events",
        "protocol": (
            "Fusion formal harness: fixed token, sequential cache, no argmax; "
            "explicit decode-horizon sweep"),
        "cuda_graph": args.cuda_graph,
        "measurements": [],
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    for batch in parse_ints(args.batches):
        for context_len in parse_ints(args.context_lengths):
            for steps in decode_steps:
                measurement = benchmark_shape(
                    model, tokenizer, args, batch, context_len, steps)
                results["measurements"].append(measurement)
                print(json.dumps(measurement, indent=2), flush=True)
                output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
