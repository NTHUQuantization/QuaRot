#!/usr/bin/env python3
"""Greedy text generation for QuaRot INT4 or original FP16 models."""

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from e2e.benchmark_real import load_int4
except ImportError:
    from benchmark_real import load_int4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        help="Path to a converted QuaRot INT4 checkpoint.",
    )
    model_group.add_argument(
        "--fp16-model",
        help="Original Hugging Face FP16 model name or local path.",
    )
    parser.add_argument(
        "--tokenizer",
        default="meta-llama/Llama-2-7b-hf",
        help="Tokenizer name or local path (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt",
        default="The capital of France is",
        help="Prompt to complete (default: %(default)r).",
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=16,
        help="Number of tokens to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--fp16-cache",
        action="store_true",
        help="Use the same INT4 weights with an FP16 KV cache for comparison.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download tokenizer files from Hugging Face.",
    )
    return parser.parse_args()


@torch.inference_mode()
def greedy_decode(model, input_ids, new_tokens):
    model._expected_max_length = input_ids.shape[1] + new_tokens
    generated = input_ids

    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(input_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_elapsed = time.perf_counter() - started

    past_key_values = output.past_key_values
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = torch.cat((generated, next_token), dim=1)

    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(new_tokens - 1):
        output = model(
            generated[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    torch.cuda.synchronize()
    decode_elapsed = time.perf_counter() - started

    return generated, prefill_elapsed, decode_elapsed


def main():
    args = parse_args()
    if args.new_tokens <= 0:
        raise ValueError("--new-tokens must be positive")
    if args.fp16_model and args.fp16_cache:
        raise ValueError("--fp16-cache only applies to a QuaRot INT4 model")
    if not torch.cuda.is_available():
        raise RuntimeError("QuaRot INT4 generation requires a CUDA/ROCm GPU")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=args.local_files_only,
    )
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()

    if args.fp16_model:
        model = AutoModelForCausalLM.from_pretrained(
            args.fp16_model,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            local_files_only=args.local_files_only,
        ).eval().cuda()
        cache_dtype = "float16"
        model_name = args.fp16_model
    else:
        model = load_int4(args.model).eval().cuda()
        if args.fp16_cache:
            model.cache_dtype = "float16"
        cache_dtype = model.cache_dtype
        model_name = args.model

    generated, prefill_elapsed, decode_elapsed = greedy_decode(
        model, input_ids, args.new_tokens)
    new_token_ids = generated[0, input_ids.shape[1] :].tolist()
    prefill_tokens = input_ids.numel()
    decode_tokens = args.new_tokens - 1
    result = {
        "model": model_name,
        "weight_dtype": "float16" if args.fp16_model else "int4",
        "cache_dtype": cache_dtype,
        "prompt": args.prompt,
        "completion": tokenizer.decode(new_token_ids, skip_special_tokens=True),
        "generated_tokens": args.new_tokens,
        "prefill_tokens": prefill_tokens,
        "prefill_seconds": prefill_elapsed,
        "prefill_tokens_per_second": prefill_tokens / prefill_elapsed,
        "decode_tokens": decode_tokens,
        "decode_seconds": decode_elapsed,
        "decode_seconds_per_token": (
            decode_elapsed / decode_tokens if decode_tokens else None),
        "decode_tokens_per_second": (
            decode_tokens / decode_elapsed if decode_tokens else None),
        "total_generation_seconds": prefill_elapsed + decode_elapsed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
