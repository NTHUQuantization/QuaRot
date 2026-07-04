#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from llama31_quarot.common import (
    RuntimeConfig,
    cuda_event_time_ms,
    load_prompts,
    load_tokenizer_and_model,
    model_shape_summary,
    summarize,
    write_csv,
)
from llama31_quarot.hf_quarot_model import wrap_model
from llama31_quarot.model_patch import inspect_model_for_quarot, require_fused_full_model_supported


def parse_ints(text):
    return [int(x) for x in text.replace(",", " ").split()]


def make_inputs(tokenizer, prompt, batch, context_len, device):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = encoded.input_ids[0]
    if ids.numel() < context_len:
        pad = ids[-1:].repeat(context_len - ids.numel())
        ids = torch.cat([ids, pad], dim=0)
    else:
        ids = ids[:context_len]
    input_ids = ids.unsqueeze(0).repeat(batch, 1).to(device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@torch.inference_mode()
def prefill(model, inputs):
    return model(**inputs, use_cache=True)


@torch.inference_mode()
def decode_one(model, next_input_ids, attention_mask, past_key_values):
    return model(
        input_ids=next_input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )


def benchmark_mode(model, tokenizer, args, out_dir):
    rows = []
    stat_rows = []
    prompts = load_prompts(args.prompts)
    prompt = prompts[0]
    wrapper = wrap_model(model, args.mode)
    for batch in parse_ints(args.batches):
        for context_len in parse_ints(args.context_lengths):
            inputs = make_inputs(tokenizer, prompt, batch, context_len, args.device)
            next_ids = inputs["input_ids"][:, -1:]

            prefill_ms_repeats = []
            decode_ms_repeats = []
            for repeat in range(args.repeats):
                torch.cuda.empty_cache()
                decode_capacity = args.warmup + args.iters + 1
                prefill_ms = cuda_event_time_ms(
                    lambda: wrapper.prefill(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=decode_capacity,
                    ),
                    args.warmup,
                    args.iters,
                )
                _, cache = wrapper.prefill(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=decode_capacity,
                )
                torch.cuda.synchronize()
                decode_ms = cuda_event_time_ms(
                    lambda: wrapper.decode_one(next_ids, cache),
                    args.warmup,
                    args.iters,
                )
                prefill_ms_repeats.append(prefill_ms)
                decode_ms_repeats.append(decode_ms)
                rows.append({
                    "mode": args.mode,
                    "batch": batch,
                    "context_len": context_len,
                    "repeat": repeat,
                    "prefill_ms": prefill_ms,
                    "decode_ms_per_token": decode_ms,
                    "decode_tokens_per_sec": 1000.0 * batch / decode_ms,
                })
            for name, values in [("prefill_ms", prefill_ms_repeats), ("decode_ms_per_token", decode_ms_repeats)]:
                s = summarize(values)
                stat_rows.append({
                    "mode": args.mode,
                    "batch": batch,
                    "context_len": context_len,
                    "metric": name,
                    **s,
                })
    write_csv(out_dir / "latency_repeats.csv", rows)
    write_csv(out_dir / "latency_summary.csv", stat_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--mode", choices=["fp16_hf", "quarot_unfused", "fused_quarot"], default="fp16_hf")
    parser.add_argument("--out-dir", default="llama31_full_model_results")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--context-lengths", default="10,128,1024")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = RuntimeConfig(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        seed=args.seed,
        local_files_only=args.local_files_only,
    )
    tokenizer, model = load_tokenizer_and_model(cfg)
    status = inspect_model_for_quarot(model, args.model_id)
    (out_dir / "model_shape.json").write_text(json.dumps(status.__dict__, indent=2))
    (out_dir / "hf_config_summary.json").write_text(json.dumps(model_shape_summary(model), indent=2))

    if args.mode != "fp16_hf":
        require_fused_full_model_supported(model)
    benchmark_mode(model, tokenizer, args, out_dir)


if __name__ == "__main__":
    main()
