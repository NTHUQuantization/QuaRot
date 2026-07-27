#!/usr/bin/env python3
import argparse
import csv
import ctypes
import json
import math
import os
import platform
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

from llama31_quarot.benchmark_full_model import make_inputs, parse_ints
from llama31_quarot.common import (
    RuntimeConfig,
    load_prompts,
    load_tokenizer_and_model,
    logits_metrics,
    model_shape_summary,
    summarize,
    write_csv,
)
from llama31_quarot.hf_quarot_model import FusionConfig, wrap_model
from llama31_quarot.model_patch import inspect_model_for_quarot, require_fused_full_model_supported


VARIANT_CONFIGS = {
    "unfused_INT4": FusionConfig.from_name("unfused_INT4"),
    "fused_current": FusionConfig.from_name("full_fused_current"),
    "fused_hadacore256": FusionConfig.from_name("full_fused_hadacore256"),
}

ABLATION_CONFIGS = {
    "unfused_INT4": FusionConfig.from_name("unfused_INT4"),
    "k1_fused": FusionConfig.from_name("k1_fused"),
    "k1_k2_fused": FusionConfig.from_name("k1_k2_fused"),
    "attention_fused_current": FusionConfig.from_name("attention_fused_current"),
    "full_fused_current": FusionConfig.from_name("full_fused_current"),
    "full_fused_hadacore256": FusionConfig.from_name("full_fused_hadacore256"),
}

STAGE_CATEGORIES = {
    "embedding_rotary_setup": "other",
    "input_rmsnorm": "norm_residual",
    "q_proj": "projection",
    "k_proj": "projection",
    "v_proj": "projection",
    "rope": "attention",
    "qkv_layout": "projection",
    "k1_unfused": "quantization",
    "k1_fused": "quantization",
    "k1_fp16_append": "attention",
    "k2_cache_dequant": "quantization",
    "k2_fp16_decode": "attention",
    "k2_int4_decode": "attention",
    "k3_unfused": "quantization",
    "k3_fused": "quantization",
    "k3_dequant": "quantization",
    "o_proj_residual": "projection",
    "post_attention_rmsnorm": "norm_residual",
    "gate_proj": "projection",
    "up_proj": "projection",
    "ffn_unfused": "quantization",
    "ffn_fused": "quantization",
    "ffn_fp16": "other",
    "ffn_dequant": "quantization",
    "down_proj_residual": "projection",
    "final_norm_lm_head": "projection",
}


class HipEventStageTimer:
    def __init__(self):
        self.records = []

    @contextmanager
    def stage(self, name, layer_idx=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.records.append((name, layer_idx, start, end))

    def rows(self, variant, batch, context_len, iterations, num_layers, plain_ms, instrumented_e2e_ms):
        torch.cuda.synchronize()
        totals = defaultdict(float)
        layer_records = defaultdict(int)
        for name, layer_idx, start, end in self.records:
            totals[name] += start.elapsed_time(end)
            if layer_idx is not None:
                layer_records[name] += 1
        stage_sum_ms = sum(totals.values()) / iterations
        output = []
        for name in sorted(totals):
            total_ms = totals[name] / iterations
            per_layer_ms = total_ms / num_layers if layer_records[name] else total_ms
            output.append({
                "variant": variant,
                "batch": batch,
                "context_len": context_len,
                "stage": name,
                "category": STAGE_CATEGORIES.get(name, "other"),
                "total_ms_per_token": total_ms,
                "per_layer_ms": per_layer_ms,
                "percent_of_stage_sum": 100.0 * total_ms / stage_sum_ms if stage_sum_ms else 0.0,
                "stage_sum_ms_per_token": stage_sum_ms,
                "instrumented_ms_per_token": instrumented_e2e_ms,
                "plain_ms_per_token": plain_ms,
                "instrumentation_overhead_percent": (
                    100.0 * (instrumented_e2e_ms - plain_ms) / plain_ms if plain_ms else 0.0
                ),
            })
        return output


class RoctxController:
    def __init__(self):
        try:
            self.lib = ctypes.CDLL("librocprofiler-sdk-roctx.so")
        except OSError:
            self.lib = ctypes.CDLL("libroctx64.so")
        self.lib.roctxProfilerPause.argtypes = [ctypes.c_uint64]
        self.lib.roctxProfilerResume.argtypes = [ctypes.c_uint64]
        self.lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        self.lib.roctxRangePop.argtypes = []

    def pause(self):
        self.lib.roctxProfilerPause(0)

    def resume(self):
        self.lib.roctxProfilerResume(0)

    def push(self, label):
        self.lib.roctxRangePushA(label.encode("utf-8"))

    def pop(self):
        self.lib.roctxRangePop()


def parse_names(text):
    return [x for x in text.replace(",", " ").split() if x]


def event_time_ms(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


@torch.inference_mode()
def hf_prefill(model, inputs):
    return model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        use_cache=True,
        logits_to_keep=1,
    )


def build_cache(wrapper, past_key_values, batch, context_len, max_new_tokens):
    return wrapper._make_cache(past_key_values, batch, context_len, max_new_tokens)


def fixed_latency(wrapper, next_ids, cache, config, warmup, iterations):
    return event_time_ms(
        lambda: wrapper.decode_one(
            next_ids,
            cache,
            advance_cache=False,
            fusion_config=config,
        ),
        warmup,
        iterations,
    )


def run_latency_grid(model, tokenizer, args, out_dir):
    rows = []
    variants = parse_names(args.variants)
    prompt = load_prompts(args.prompts)[0]
    for session in range(args.sessions):
        for batch in parse_ints(args.batches):
            for context_len in parse_ints(args.context_lengths):
                inputs = make_inputs(tokenizer, prompt, batch, context_len, args.device)
                next_ids = inputs["input_ids"][:, -1:]
                prefill_out = hf_prefill(model, inputs)
                order = variants[session % len(variants):] + variants[:session % len(variants)]
                for variant in order:
                    config = VARIANT_CONFIGS[variant]
                    wrapper = wrap_model(
                        model,
                        "fused_quarot",
                        fusion_backend=config.backend,
                        fusion_config=config,
                    )
                    cache = build_cache(wrapper, prefill_out.past_key_values, batch, context_len, 1)
                    for repeat in range(args.repeats):
                        latency_ms = fixed_latency(
                            wrapper,
                            next_ids,
                            cache,
                            config,
                            args.warmup,
                            args.iters,
                        )
                        rows.append({
                            "session": session,
                            "repeat": repeat,
                            "variant": variant,
                            "batch": batch,
                            "context_len": context_len,
                            "warmup": args.warmup,
                            "iterations": args.iters,
                            "decode_ms_per_token": latency_ms,
                            "tokens_per_second": 1000.0 * batch / latency_ms,
                        })
                    del cache
                    torch.cuda.empty_cache()
                    write_csv(out_dir / "latency_raw.csv", rows)
                del prefill_out
                torch.cuda.empty_cache()
    return rows


def run_correctness(model, tokenizer, args, out_dir):
    rows = []
    prompt = load_prompts(args.prompts)[0]
    for batch, context_len in representative_shapes(args):
        inputs = make_inputs(tokenizer, prompt, batch, context_len, args.device)
        next_ids = inputs["input_ids"][:, -1:]
        prefill_out = hf_prefill(model, inputs)
        outputs = {}
        for variant, config in VARIANT_CONFIGS.items():
            wrapper = wrap_model(model, "fused_quarot", config.backend, config)
            cache = build_cache(wrapper, prefill_out.past_key_values, batch, context_len, 1)
            seq_before = cache.seq_len
            logits, _ = wrapper.decode_one(
                next_ids,
                cache,
                advance_cache=False,
                fusion_config=config,
            )
            torch.cuda.synchronize()
            if cache.seq_len != seq_before:
                raise AssertionError("fixed-context decode changed cache.seq_len")
            outputs[variant] = logits.detach().float().cpu()
            del cache
        reference = outputs["unfused_INT4"]
        for variant in ["fused_current", "fused_hadacore256"]:
            metrics = logits_metrics(reference, outputs[variant], topk=10)
            rows.append({
                "variant": variant,
                "reference": "unfused_INT4",
                "batch": batch,
                "context_len": context_len,
                **metrics,
                "has_nan_or_inf": int(not torch.isfinite(outputs[variant]).all().item()),
            })
        del prefill_out
        torch.cuda.empty_cache()
        write_csv(out_dir / "correctness.csv", rows)
    return rows


def run_stage_breakdown(model, tokenizer, args, out_dir):
    rows = []
    prompt = load_prompts(args.prompts)[0]
    for batch, context_len in representative_shapes(args):
        inputs = make_inputs(tokenizer, prompt, batch, context_len, args.device)
        next_ids = inputs["input_ids"][:, -1:]
        prefill_out = hf_prefill(model, inputs)
        for variant, config in VARIANT_CONFIGS.items():
            wrapper = wrap_model(model, "fused_quarot", config.backend, config)
            cache = build_cache(wrapper, prefill_out.past_key_values, batch, context_len, 1)
            plain_ms = fixed_latency(
                wrapper,
                next_ids,
                cache,
                config,
                args.stage_warmup,
                args.stage_iters,
            )
            timer = HipEventStageTimer()
            instrumented_start = torch.cuda.Event(enable_timing=True)
            instrumented_end = torch.cuda.Event(enable_timing=True)
            instrumented_start.record()
            for _ in range(args.stage_iters):
                wrapper.decode_one(
                    next_ids,
                    cache,
                    advance_cache=False,
                    stage_timer=timer,
                    fusion_config=config,
                )
            instrumented_end.record()
            torch.cuda.synchronize()
            instrumented_e2e_ms = instrumented_start.elapsed_time(instrumented_end) / args.stage_iters
            rows.extend(timer.rows(
                variant,
                batch,
                context_len,
                args.stage_iters,
                model.config.num_hidden_layers,
                plain_ms,
                instrumented_e2e_ms,
            ))
            del cache
            torch.cuda.empty_cache()
            write_csv(out_dir / "stage_breakdown.csv", rows)
        del prefill_out
        torch.cuda.empty_cache()
    return rows


def run_ablation(model, tokenizer, args, out_dir):
    rows = []
    prompt = load_prompts(args.prompts)[0]
    for batch, context_len in representative_shapes(args):
        inputs = make_inputs(tokenizer, prompt, batch, context_len, args.device)
        next_ids = inputs["input_ids"][:, -1:]
        prefill_out = hf_prefill(model, inputs)
        for variant, config in ABLATION_CONFIGS.items():
            wrapper = wrap_model(model, "fused_quarot", config.backend, config)
            cache = build_cache(wrapper, prefill_out.past_key_values, batch, context_len, 1)
            for repeat in range(args.ablation_repeats):
                latency_ms = fixed_latency(
                    wrapper,
                    next_ids,
                    cache,
                    config,
                    args.ablation_warmup,
                    args.ablation_iters,
                )
                rows.append({
                    "repeat": repeat,
                    "variant": variant,
                    "batch": batch,
                    "context_len": context_len,
                    "decode_ms_per_token": latency_ms,
                })
            del cache
            torch.cuda.empty_cache()
            write_csv(out_dir / "ablation_raw.csv", rows)
        del prefill_out
        torch.cuda.empty_cache()
    return rows


def run_sequential(model, tokenizer, args, out_dir):
    rows = []
    prompt = load_prompts(args.prompts)[0]
    shapes = [(1, 128), (1, 4096)]
    for batch, context_len in shapes:
        inputs = make_inputs(tokenizer, prompt, batch, context_len, args.device)
        prefill_out = hf_prefill(model, inputs)
        first_ids = inputs["input_ids"][:, -1:]
        for variant, config in VARIANT_CONFIGS.items():
            wrapper = wrap_model(model, "fused_quarot", config.backend, config)
            cache = build_cache(
                wrapper,
                prefill_out.past_key_values,
                batch,
                context_len,
                args.sequential_tokens,
            )
            next_ids = first_ids
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(args.sequential_tokens):
                logits, cache = wrapper.decode_one(
                    next_ids,
                    cache,
                    advance_cache=True,
                    fusion_config=config,
                )
                next_ids = logits.argmax(dim=-1, keepdim=True)
            end.record()
            torch.cuda.synchronize()
            total_ms = start.elapsed_time(end)
            rows.append({
                "variant": variant,
                "batch": batch,
                "initial_context_len": context_len,
                "tokens": args.sequential_tokens,
                "total_ms": total_ms,
                "decode_ms_per_token": total_ms / args.sequential_tokens,
                "tokens_per_second": 1000.0 * batch * args.sequential_tokens / total_ms,
            })
            del cache
            torch.cuda.empty_cache()
            write_csv(out_dir / "sequential_decode.csv", rows)
        del prefill_out
        torch.cuda.empty_cache()
    return rows


def run_rocprof_workload(model, tokenizer, args, out_dir, roctx):
    variant = args.rocprof_variant
    config = VARIANT_CONFIGS[variant]
    prompt = load_prompts(args.prompts)[0]
    inputs = make_inputs(tokenizer, prompt, args.rocprof_batch, args.rocprof_context, args.device)
    next_ids = inputs["input_ids"][:, -1:]
    prefill_out = hf_prefill(model, inputs)
    wrapper = wrap_model(model, "fused_quarot", config.backend, config)
    cache = build_cache(
        wrapper,
        prefill_out.past_key_values,
        args.rocprof_batch,
        args.rocprof_context,
        1,
    )
    for _ in range(args.rocprof_warmup):
        wrapper.decode_one(next_ids, cache, advance_cache=False, fusion_config=config)
    torch.cuda.synchronize()
    roctx.resume()
    roctx.push(f"decode_only/{variant}/B{args.rocprof_batch}/L{args.rocprof_context}")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.rocprof_iters):
        wrapper.decode_one(next_ids, cache, advance_cache=False, fusion_config=config)
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    roctx.pop()
    roctx.pause()
    row = {
        "variant": variant,
        "batch": args.rocprof_batch,
        "context_len": args.rocprof_context,
        "iterations": args.rocprof_iters,
        "total_ms": elapsed_ms,
        "decode_ms_per_token": elapsed_ms / args.rocprof_iters,
    }
    write_csv(out_dir / f"{variant}_B{args.rocprof_batch}_L{args.rocprof_context}_event.csv", [row])
    return [row]


def representative_shapes(args):
    values = []
    for item in args.representative_shapes.split(","):
        batch, context = item.lower().split("x")
        values.append((int(batch), int(context)))
    return values


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={Path.cwd()}", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_environment(out_dir, model):
    properties = torch.cuda.get_device_properties(0)
    device_name = torch.cuda.get_device_name(0) or getattr(properties, "gcnArchName", "unknown")
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": device_name,
        "device_properties": str(properties),
        "model": model_shape_summary(model),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_short": git_output("status", "--short").splitlines(),
        "hf_token_recorded": False,
    }
    (out_dir / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["all", "latency", "correctness", "stage", "ablation", "sequential", "rocprof_workload"],
        default="all",
    )
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--out-dir", default="decode_bottleneck_profiling_results")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--context-lengths", default="10,128,1024,4096")
    parser.add_argument("--variants", default="unfused_INT4,fused_current,fused_hadacore256")
    parser.add_argument("--representative-shapes", default="1x128,1x4096,4x1024")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--stage-warmup", type=int, default=3)
    parser.add_argument("--stage-iters", type=int, default=3)
    parser.add_argument("--ablation-warmup", type=int, default=5)
    parser.add_argument("--ablation-iters", type=int, default=20)
    parser.add_argument("--ablation-repeats", type=int, default=3)
    parser.add_argument("--sequential-tokens", type=int, default=32)
    parser.add_argument("--rocprof-variant", choices=sorted(VARIANT_CONFIGS), default="fused_current")
    parser.add_argument("--rocprof-batch", type=int, default=1)
    parser.add_argument("--rocprof-context", type=int, default=128)
    parser.add_argument("--rocprof-warmup", type=int, default=3)
    parser.add_argument("--rocprof-iters", type=int, default=10)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roctx = None
    if args.task == "rocprof_workload":
        roctx = RoctxController()

    runtime = RuntimeConfig(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
        local_files_only=args.local_files_only,
    )
    tokenizer, model = load_tokenizer_and_model(runtime)
    status = inspect_model_for_quarot(model, args.model_id)
    require_fused_full_model_supported(model)
    (out_dir / "model_shape.json").write_text(json.dumps(status.__dict__, indent=2) + "\n")
    write_environment(out_dir, model)

    if args.task in {"all", "correctness"}:
        run_correctness(model, tokenizer, args, out_dir)
    if args.task in {"all", "latency"}:
        run_latency_grid(model, tokenizer, args, out_dir)
    if args.task in {"all", "stage"}:
        run_stage_breakdown(model, tokenizer, args, out_dir)
    if args.task in {"all", "ablation"}:
        run_ablation(model, tokenizer, args, out_dir)
    if args.task in {"all", "sequential"}:
        run_sequential(model, tokenizer, args, out_dir)
    if args.task == "rocprof_workload":
        run_rocprof_workload(model, tokenizer, args, out_dir, roctx)


if __name__ == "__main__":
    main()
