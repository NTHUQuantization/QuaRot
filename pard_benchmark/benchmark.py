from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from dataclasses import asdict, replace
from pathlib import Path

from .config import MODEL_SPECS, UPSTREAM_COMMIT, hf_token, load_hf_env


DEFAULT_PROMPT = (
    "Efficient language-model inference requires careful control of cache layout, "
    "memory traffic, batching, and numerical correctness. Explain the trade-offs "
    "and give a concrete example."
)


def make_context(tokenizer, prompt: str, context_len: int, device):
    import torch

    ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
    if ids.numel() == 0:
        raise ValueError("tokenizer produced an empty prompt")
    repeats = (context_len + ids.numel() - 1) // ids.numel()
    ids = ids.repeat(repeats)[:context_len]
    return ids.unsqueeze(0).to(device=device, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone PARD/PARD2 Llama 3.1 decode benchmark")
    parser.add_argument("--mode", choices=["ar", "pard", "pard2-ti", "pard2-td"], required=True)
    parser.add_argument("--target", choices=["base", "instruct"], default="base")
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--w4a4-checkpoint", default=None)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--draft-revision", default=None)
    parser.add_argument("--draft-k", type=int, default=12)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile-mode", choices=["eager", "reduce-overhead", "max-autotune"], default="eager")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--phase", choices=["smoke", "tune", "formal", "manual"], default="manual")
    parser.add_argument("--out-dir", default="pard_decode_results")
    parser.add_argument("--result-name", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--stop-on-eos", action="store_true", help="Stop early at EOS instead of measuring a fixed token count")
    parser.add_argument("--hf-env", default=".hf_env")
    return parser.parse_args()


def resolve_specs(args):
    target_spec = MODEL_SPECS[args.target]
    if args.target_model:
        target_spec = replace(target_spec, model_id=args.target_model)
    if args.target_revision:
        target_spec = replace(target_spec, revision=args.target_revision)
    draft_spec = None
    if args.mode == "pard":
        draft_spec = MODEL_SPECS["pard"]
    elif args.mode.startswith("pard2"):
        draft_spec = MODEL_SPECS["pard2"]
    if draft_spec and args.draft_model:
        draft_spec = replace(draft_spec, model_id=args.draft_model)
    if draft_spec and args.draft_revision:
        draft_spec = replace(draft_spec, revision=args.draft_revision)
    return target_spec, draft_spec


def result_filename(args) -> str:
    if args.result_name:
        return args.result_name
    mode = args.mode.replace("-", "_")
    return (
        f"{args.phase}_{args.target}_{mode}_k{args.draft_k}_"
        f"l{args.context_len}_{args.compile_mode.replace('-', '_')}.json"
    )


def main():
    args = parse_args()
    if args.draft_k < 1:
        raise ValueError("draft-k must be positive")
    load_hf_env(args.hf_env)
    # Import only after HF_HOME has been loaded.
    import torch
    import transformers

    from .engine import load_runtime
    from .w4a4_engine import load_w4a4_runtime
    from .measurement import environment_manifest, gpu_snapshot

    torch.manual_seed(42)
    if args.compile_mode != "eager":
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._dynamo.config.cache_size_limit = 128
    target_spec, draft_spec = resolve_specs(args)
    max_cache_len = args.context_len + args.max_new_tokens + args.draft_k + 32
    if args.w4a4_checkpoint:
        if args.mode not in {"ar", "pard", "pard2-ti"}:
            raise ValueError("packed W4A4 target supports ar, pard, and pard2-ti")
        if args.compile_mode != "eager":
            raise ValueError("packed W4A4 target currently requires --compile-mode eager")
        runtime, compatibility = load_w4a4_runtime(
            checkpoint_path=args.w4a4_checkpoint,
            draft_spec=draft_spec,
            tokenizer_spec=target_spec,
            draft_k=args.draft_k,
            max_cache_len=max_cache_len,
            local_files_only=args.local_files_only,
            token=hf_token(),
            stop_on_eos=args.stop_on_eos,
        )
    else:
        runtime, compatibility = load_runtime(
            mode=args.mode,
            target_spec=target_spec,
            draft_spec=draft_spec,
            draft_k=args.draft_k,
            max_cache_len=max_cache_len,
            compile_mode=args.compile_mode,
            local_files_only=args.local_files_only,
            token=hf_token(),
            stop_on_eos=args.stop_on_eos,
        )
    inputs = make_context(runtime.tokenizer, args.prompt, args.context_len, "cuda")
    for _ in range(args.warmups):
        runtime.generate(inputs, args.max_new_tokens, capture_memory=False)
        gc.collect()
        torch.cuda.empty_cache()
    repeats = []
    for repeat in range(args.repeats):
        gc.collect()
        torch.cuda.empty_cache()
        result = runtime.generate(inputs, args.max_new_tokens, capture_memory=True)
        repeats.append(
            {
                "repeat": repeat,
                **result.metrics(),
                "output_ids": result.output_ids,
                "memory": result.memory,
            }
        )
    speeds = [x["steady_tokens_per_s"] for x in repeats]
    payload = {
        "schema_version": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "phase": args.phase,
        "mode": args.mode,
        "runtime_kind": "quarot_w4a4kv4" if args.w4a4_checkpoint else "hf",
        "w4a4_checkpoint": str(Path(args.w4a4_checkpoint).resolve()) if args.w4a4_checkpoint else None,
        "target_key": args.target,
        "target": asdict(target_spec),
        "draft": asdict(draft_spec) if draft_spec else None,
        "draft_k": args.draft_k,
        "context_len": args.context_len,
        "max_new_tokens": args.max_new_tokens,
        "warmups": args.warmups,
        "repeat_count": args.repeats,
        "compile_mode": args.compile_mode,
        "attention_implementation": "sdpa",
        "dtype": "w4a4kv4-fp16-acc32" if args.w4a4_checkpoint else "bfloat16",
        "stop_on_eos": args.stop_on_eos,
        "compatibility": compatibility.to_dict(),
        "environment": environment_manifest(torch, transformers),
        "parameter_bytes": {
            "target": int(runtime.target.parameter_bytes),
            "draft": int(runtime.draft.parameter_bytes) if runtime.draft is not None else 0,
        },
        "load_memory": runtime.load_memory,
        "repeats": repeats,
        "summary": {
            "steady_tokens_per_s_median": statistics.median(speeds),
            "steady_tokens_per_s_mean": statistics.mean(speeds),
            "final_memory": gpu_snapshot(torch, "result_write"),
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / result_filename(args)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(out_path), "median_steady_tokens_per_s": statistics.median(speeds)}))


if __name__ == "__main__":
    main()
