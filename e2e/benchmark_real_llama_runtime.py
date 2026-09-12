"""Benchmark real INT4 and FP16 checkpoints with the direct Llama runtime.

The shared workload is portable to older Llama-only QuaRot checkouts.
"""
import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import quarot
from e2e.quantized_llama import modeling_llama

def load_real_int4_model(checkpoint):
    config = modeling_llama.QuarotLlamaConfig.from_pretrained(
        checkpoint, attn_implementation="flash_attention_2",
        local_files_only=True)
    return modeling_llama.QuarotLlamaForCausalLM.from_pretrained(
        checkpoint, config=config, torch_dtype=torch.float16,
        local_files_only=True)


def load_fp16_model(checkpoint):
    return modeling_llama.QuarotFP16LlamaForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )


def validate_real_int4_checkpoint(model):
    """Reject an uninitialized or partially loaded INT4 model."""
    modules = [module for module in model.modules()
               if isinstance(module, quarot.nn.Linear4bit)]
    if not modules:
        raise RuntimeError("loaded model contains no Linear4bit modules")
    for module in modules:
        if module.weight.dtype != torch.uint8 or module.weight.numel() == 0:
            raise RuntimeError("invalid packed INT4 weight in checkpoint")
        scales = module.weight_scales
        if (not torch.isfinite(scales).all() or
                torch.count_nonzero(scales) != scales.numel()):
            raise RuntimeError("invalid INT4 weight scales in checkpoint")
    return len(modules)


# Keep all measurement code below identical in every compared checkout.
def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def gpu_name():
    name = torch.cuda.get_device_name(0)
    if name:
        return name
    return getattr(torch.cuda.get_device_properties(0),
                   "gcnArchName", "unknown")


def deterministic_tokens(batch_size, length, vocab_size, device):
    values = torch.arange(
        batch_size * length, dtype=torch.long, device=device)
    return values.reshape(batch_size, length).remainder(
        max(vocab_size - 3, 1)).add_(3)


def percentile(ordered, fraction):
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return (ordered[lower] * (upper - position) +
            ordered[upper] * (position - lower))


def summarize(samples_ms):
    ordered = sorted(float(value) for value in samples_ms)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "stddev_ms": statistics.pstdev(ordered),
        "p10_ms": percentile(ordered, 0.10),
        "p90_ms": percentile(ordered, 0.90),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": ordered,
    }


@torch.inference_mode()
def measure(workload, warmup, repeats):
    """Measure synchronized wall time and peak allocated CUDA memory.

    Each timing sample is one complete workload execution.
    Memory is the highest allocated memory observed across all measured runs.
    It includes model weights and any persistent cache owned by the workload.
    ``incremental_peak_memory_bytes`` excludes allocations already live just
    before measurement (normally weights plus a prepared decode cache).
    """
    for _ in range(warmup):
        workload()
    torch.cuda.synchronize()

    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        workload()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    result = summarize(samples)
    peak = torch.cuda.max_memory_allocated()
    result["baseline_memory_bytes"] = baseline
    result["peak_memory_bytes"] = peak
    result["incremental_peak_memory_bytes"] = max(peak - baseline, 0)
    return result


def make_prefill(model, prompt, max_length):
    def workload():
        model._expected_max_length = max_length
        model(prompt, use_cache=True)
    return workload


@torch.inference_mode()
def make_decode(model, prompt, next_token, max_length):
    model._expected_max_length = max_length
    output = model(prompt, use_cache=True)
    cache = output.past_key_values
    del output

    def workload():
        cache.length = prompt.shape[1]
        for _ in range(max_length - prompt.shape[1]):
            model(next_token, past_key_values=cache, use_cache=True)
    return workload


def make_e2e(model, prompt, next_token, max_length):
    def workload():
        model._expected_max_length = max_length
        output = model(prompt, use_cache=True)
        cache = output.past_key_values
        for _ in range(max_length - prompt.shape[1]):
            model(next_token, past_key_values=cache, use_cache=True)
    return workload


def benchmark_one(name, factory, args):
    cleanup()
    workload = factory()
    result = measure(workload, args.warmup, args.repeats)
    del workload
    cleanup()
    print("{}: median {:.3f} ms, peak {:.3f} GiB".format(
        name, result["median_ms"],
        result["peak_memory_bytes"] / (1024 ** 3)), flush=True)
    return result


def parse_args(description=None, default_output="benchmark_real_llama_runtime.json"):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("--int4-model", required=True,
                        help="Path to a real converted QuaRot INT4 checkpoint")
    parser.add_argument("--fp16-model",
                        help="Path or Hugging Face ID of the FP16 base model")
    parser.add_argument(
        "--int4-only", action="store_true",
        help="Benchmark only INT4; skip FP16 measurements and speedups")
    parser.add_argument(
        "--kv-cache-dtype", choices=("int4", "float16"), default="int4",
        help="KV-cache dtype for the INT4 model (default: int4)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-seq-len", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args()
    if not args.int4_only and not args.fp16_model:
        parser.error("--fp16-model is required unless --int4-only is used")
    positive = (args.batch_size, args.prefill_seq_len, args.decode_steps,
                args.repeats)
    if min(positive) <= 0 or args.warmup < 0:
        parser.error("sizes, repeats, and decode steps must be positive")
    return args


def main(int4_loader=load_real_int4_model, fp16_loader=load_fp16_model,
         description=None,
         default_output="benchmark_real_llama_runtime.json"):
    args = parse_args(description, default_output)
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA or ROCm")

    torch.manual_seed(0)
    int4_model = int4_loader(args.int4_model).cuda().eval()
    int4_model.cache_dtype = args.kv_cache_dtype
    linear4bit_modules = validate_real_int4_checkpoint(int4_model)
    device = next(int4_model.parameters()).device
    prompt = deterministic_tokens(
        args.batch_size, args.prefill_seq_len,
        int4_model.config.vocab_size, device)
    next_token = torch.full(
        (args.batch_size, 1), 100, dtype=torch.long, device=device)
    max_length = args.prefill_seq_len + args.decode_steps

    results = {
        "int4_checkpoint": str(Path(args.int4_model).resolve()),
        "fp16_model": args.fp16_model,
        "linear4bit_modules": linear4bit_modules,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "hip": torch.version.hip,
            "gpu": gpu_name(),
        },
        "configuration": vars(args),
    }
    results["int4"] = {
        "prefill": benchmark_one(
            "INT4 prefill",
            lambda: make_prefill(int4_model, prompt, max_length), args),
        "decode": benchmark_one(
            "INT4 decode",
            lambda: make_decode(
                int4_model, prompt, next_token, max_length), args),
        "e2e": benchmark_one(
            "INT4 e2e",
            lambda: make_e2e(
                int4_model, prompt, next_token, max_length), args),
    }
    del int4_model
    cleanup()

    if not args.int4_only:
        fp16_model = fp16_loader(args.fp16_model).cuda().eval()
        # The reference always uses FP16 KV, independent of the INT4 cache
        # selection, so its configuration cannot depend on model defaults.
        fp16_model.cache_dtype = "float16"
        results["fp16"] = {
            "prefill": benchmark_one(
                "FP16 prefill",
                lambda: make_prefill(fp16_model, prompt, max_length), args),
            "decode": benchmark_one(
                "FP16 decode",
                lambda: make_decode(
                    fp16_model, prompt, next_token, max_length), args),
            "e2e": benchmark_one(
                "FP16 e2e",
                lambda: make_e2e(
                    fp16_model, prompt, next_token, max_length), args),
        }
        del fp16_model
        cleanup()

        results["speedup"] = {
            name: (results["fp16"][name]["median_ms"] /
                   results["int4"][name]["median_ms"])
            for name in ("prefill", "decode", "e2e")
        }
        results["peak_memory_reduction"] = {
            name: (results["fp16"][name]["peak_memory_bytes"] /
                   results["int4"][name]["peak_memory_bytes"])
            for name in ("prefill", "decode", "e2e")
        }
        print("speedup: prefill={:.3f}x decode={:.3f}x e2e={:.3f}x".format(
            results["speedup"]["prefill"], results["speedup"]["decode"],
            results["speedup"]["e2e"]), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print("wrote {}".format(output), flush=True)


if __name__ == "__main__":
    main()
