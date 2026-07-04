#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from llama31_quarot.common import (
    RuntimeConfig,
    load_prompts,
    load_tokenizer_and_model,
    logits_metrics,
    write_csv,
)
from llama31_quarot.hf_quarot_model import wrap_model
from llama31_quarot.model_patch import inspect_model_for_quarot, require_fused_full_model_supported


def encode_prompt(tokenizer, prompt, max_length, device):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    return {k: v.to(device) for k, v in encoded.items()}


@torch.inference_mode()
def run_logits(model, inputs):
    return model(**inputs, use_cache=False, output_hidden_states=True)


@torch.inference_mode()
def run_decode_logits(wrapper, inputs):
    _, cache = wrapper.prefill(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=1,
    )
    logits, _ = wrapper.decode_one(inputs["input_ids"][:, -1:], cache)
    return logits


@torch.inference_mode()
def generate_text(wrapper, tokenizer, inputs, max_new_tokens):
    out = wrapper.generate(
        inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--reference-mode", choices=["fp16_hf", "quarot_unfused", "fused_quarot"], default="fp16_hf")
    parser.add_argument("--candidate-mode", choices=["fp16_hf", "quarot_unfused", "fused_quarot"], default="fp16_hf")
    parser.add_argument("--out-dir", default="llama31_full_model_results")
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--max-lengths", default="16,128,1024")
    parser.add_argument("--max-new-tokens", type=int, default=32)
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
    if args.reference_mode != "fp16_hf" or args.candidate_mode != "fp16_hf":
        require_fused_full_model_supported(model)
    ref_wrapper = wrap_model(model, args.reference_mode)
    cand_wrapper = wrap_model(model, args.candidate_mode)

    prompts = load_prompts(args.prompts)
    lengths = [int(x) for x in args.max_lengths.replace(",", " ").split()]
    rows = []
    samples = []
    for prompt_id, prompt in enumerate(prompts):
        for max_len in lengths:
            inputs = encode_prompt(tokenizer, prompt, max_len, args.device)
            ref = run_decode_logits(ref_wrapper, inputs)
            cand = run_decode_logits(cand_wrapper, inputs)
            metrics = logits_metrics(ref, cand, topk=10)
            rows.append({
                "reference_mode": args.reference_mode,
                "candidate_mode": args.candidate_mode,
                "prompt_id": prompt_id,
                "max_length": max_len,
                **metrics,
            })
        inputs = encode_prompt(tokenizer, prompt, lengths[0], args.device)
        samples.append({
            "prompt_id": prompt_id,
            "prompt": prompt,
            "mode": args.candidate_mode,
            "text": generate_text(cand_wrapper, tokenizer, inputs, args.max_new_tokens),
        })
    write_csv(out_dir / "quality_logits.csv", rows)
    with (out_dir / "generation_samples.md").open("w") as f:
        f.write("# Llama-3.1 8B Generation Samples\n\n")
        for sample in samples:
            f.write(f"## Prompt {sample['prompt_id']} ({sample['mode']})\n\n")
            f.write("Prompt:\n\n")
            f.write(sample["prompt"] + "\n\n")
            f.write("Output:\n\n")
            f.write(sample["text"] + "\n\n")


if __name__ == "__main__":
    main()
