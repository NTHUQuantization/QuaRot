"""Benchmark a real QuaRot INT4 checkpoint against its FP16 base model.

Unlike benchmark.py, this runner never constructs an uninitialized INT4 model.
It requires a checkpoint produced by checkpoint_utils/quantize_llama_checkpoint.py
and validates packed weights and scales before running.
"""
import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantized_llama import modeling_llama
import quarot


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def summarize(samples):
    ordered = sorted(float(x) for x in samples)
    def percentile(p):
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)
    return {
        "samples_ms": ordered,
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "stddev_ms": statistics.pstdev(ordered),
        "p10_ms": percentile(0.10),
        "p90_ms": percentile(0.90),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


@torch.inference_mode()
def measure(fn, warmup, iterations, repeats):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0 / iterations)
    result = summarize(samples)
    result["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    return result


def deterministic_tokens(batch, length, vocab_size, device):
    values = torch.arange(batch * length, device=device, dtype=torch.long)
    return (values.reshape(batch, length) % max(vocab_size - 3, 1)) + 3


def validate_int4_checkpoint(model, checkpoint):
    modules = [m for m in model.modules() if isinstance(m, quarot.nn.Linear4bit)]
    if not modules:
        raise RuntimeError(f"{checkpoint} contains no Linear4bit modules")
    failures = []
    for name, module in model.named_modules():
        if not isinstance(module, quarot.nn.Linear4bit):
            continue
        if module.weight.dtype != torch.uint8 or module.weight.numel() == 0:
            failures.append(f"{name}: invalid packed weight")
        if not torch.isfinite(module.weight_scales).all():
            failures.append(f"{name}: nonfinite weight scales")
        if torch.count_nonzero(module.weight_scales) != module.weight_scales.numel():
            failures.append(f"{name}: zero weight scales")
    if failures:
        raise RuntimeError("invalid INT4 checkpoint:\n  " + "\n  ".join(failures[:20]))
    return {
        "linear4bit_modules": len(modules),
        "scale_min": min(float(m.weight_scales.abs().min()) for m in modules),
        "scale_max": max(float(m.weight_scales.abs().max()) for m in modules),
        "packed_min": min(int(m.weight.min()) for m in modules),
        "packed_max": max(int(m.weight.max()) for m in modules),
    }


def load_int4(path):
    config = modeling_llama.QuarotLlamaConfig.from_pretrained(
        path, attn_implementation="flash_attention_2", local_files_only=True)
    model = modeling_llama.QuarotLlamaForCausalLM.from_pretrained(
        path, config=config, torch_dtype=torch.float16, local_files_only=True)
    return model


def load_fp16(path):
    return modeling_llama.QuarotFP16LlamaForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", local_files_only=True)


@torch.inference_mode()
def numerical_snapshot(model, tokens, decode_steps):
    model._expected_max_length = tokens.shape[1] + decode_steps
    out = model(tokens, use_cache=True)
    prefill = out.logits[:, -1].float().cpu()
    decode = []
    # Fixed identical decode tokens keep the two model trajectories aligned;
    # feeding each model's argmax would confound later-step logit comparison.
    next_token = torch.full((tokens.shape[0], 1), 100,
                            device=tokens.device, dtype=torch.long)
    cache = out.past_key_values
    for _ in range(decode_steps):
        out = model(next_token, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1].float().cpu()
        decode.append(logits)
    return {"prefill": prefill, "decode": decode}


def compare_snapshots(actual, expected):
    rows = []
    for phase, aa, ee in [("prefill", actual["prefill"], expected["prefill"])]:
        rows.append(compare_logits(phase, aa, ee))
    for i, (aa, ee) in enumerate(zip(actual["decode"], expected["decode"])):
        rows.append(compare_logits(f"decode_{i + 1}", aa, ee))
    return rows


def compare_logits(name, actual, expected):
    delta = actual - expected
    denom = expected.abs().clamp_min(1e-5)
    return {
        "name": name,
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "max_rel": float((delta.abs() / denom).max()),
        "cosine": float(torch.nn.functional.cosine_similarity(actual, expected).mean()),
        "top_token_agreement": float((actual.argmax(-1) == expected.argmax(-1)).float().mean()),
        "actual_top_tokens": actual.argmax(-1).tolist(),
        "expected_top_tokens": expected.argmax(-1).tolist(),
        "nan_count": int(torch.isnan(actual).sum()),
        "inf_count": int(torch.isinf(actual).sum()),
    }


def make_workloads(model, tokens, decode_steps):
    device = tokens.device
    next_token = torch.full((tokens.shape[0], 1), 100, device=device, dtype=torch.long)

    def prefill():
        model._expected_max_length = tokens.shape[1] + decode_steps
        return model(tokens, use_cache=True)

    seed = prefill()
    cache = seed.past_key_values

    def decode():
        cache.length = tokens.shape[1]
        for _ in range(decode_steps):
            model(next_token, past_key_values=cache, use_cache=True)

    def e2e():
        model._expected_max_length = tokens.shape[1] + decode_steps
        out = model(tokens, use_cache=True)
        for _ in range(decode_steps):
            model(next_token, past_key_values=out.past_key_values, use_cache=True)

    return prefill, decode, e2e


def benchmark_model(model, args, tokens):
    model.eval().cuda()
    prefill, decode, e2e = make_workloads(model, tokens, args.decode_steps)
    return {
        "prefill": measure(prefill, args.warmup, args.iterations, args.repeats),
        "decode": measure(decode, args.warmup, args.iterations, args.repeats),
        "e2e": measure(e2e, args.warmup, args.iterations, args.repeats),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--int4-model", required=True, help="real converted QuaRot checkpoint")
    parser.add_argument("--fp16-model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--correctness-prefill", type=int, default=16)
    parser.add_argument("--correctness-decode", type=int, default=2)
    parser.add_argument("--output", default="benchmark_real_results.json")
    args = parser.parse_args()
    if min(args.batch_size, args.prefill_seq_len, args.decode_steps,
           args.iterations, args.repeats) <= 0 or args.warmup < 0:
        parser.error("sizes, iterations, repeats, and decode steps must be positive")

    torch.manual_seed(0)
    device = torch.device("cuda")
    results = {
        "environment": {
            "torch": torch.__version__, "hip": torch.version.hip,
            "gpu": (torch.cuda.get_device_name(0) or
                    torch.cuda.get_device_properties(0).gcnArchName),
            "int4_model": str(Path(args.int4_model).resolve()),
            "fp16_model": args.fp16_model,
            "fused_ffn_max_rows": os.environ.get("QUAROT_FUSED_FFN_MAX_ROWS", "64"),
        },
        "configuration": vars(args), "failures": [],
    }

    int4 = load_int4(args.int4_model)
    results["int4_checkpoint_validation"] = validate_int4_checkpoint(int4, args.int4_model)
    int4.cuda().eval()
    correctness_tokens = deterministic_tokens(args.batch_size, args.correctness_prefill,
                                               int4.config.vocab_size, device)
    int4_snapshot = numerical_snapshot(int4, correctness_tokens, args.correctness_decode)
    bench_tokens = deterministic_tokens(args.batch_size, args.prefill_seq_len,
                                         int4.config.vocab_size, device)
    results["int4"] = benchmark_model(int4, args, bench_tokens)
    del int4
    cleanup()

    fp16 = load_fp16(args.fp16_model).cuda().eval()
    fp16_snapshot = numerical_snapshot(fp16, correctness_tokens, args.correctness_decode)
    results["fp16"] = benchmark_model(fp16, args, bench_tokens)
    results["numerical_comparison"] = compare_snapshots(int4_snapshot, fp16_snapshot)
    del fp16
    cleanup()

    results["speedup"] = {
        name: results["fp16"][name]["median_ms"] / results["int4"][name]["median_ms"]
        for name in ("prefill", "decode", "e2e")
    }
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
