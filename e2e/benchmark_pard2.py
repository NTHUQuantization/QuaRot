"""Qualified single-process benchmark for one fused_v1 PARD2 mode/dataset."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.pard2 import (DEFAULT_DRAFT, DEFAULT_TARGET, DEFAULT_TOKENIZER,
                       verify_local_resources, verify_target_checkpoint)
from e2e.speculative import load_runtime


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "QuaRot/.benchmark_data/amd_pard_6f279bf"
DATASETS = {
    "humaneval": (80, "e16580cc87cac3168e59bde4ec1d0cd5cec7f31e1506c8ee1eba2ff914cf4368"),
    "gsm8k": (80, "56767ac321f1e80e3721f390c743a6a7253d4d5f50bd791339010ffcd8620371"),
    "math_500": (20, "4d8a669d746329ede429b1f4b037138ada9ae39ead6d0c34e1aa1be5349365a5"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_prompts(dataset, root=DATA_ROOT):
    expected_count, expected_hash = DATASETS[dataset]
    path = Path(root) / f"{dataset}.jsonl"
    if not path.is_file() or sha256(path) != expected_hash:
        raise RuntimeError(f"missing or modified pinned AMD dataset: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != expected_count:
        raise RuntimeError(f"{dataset} must contain exactly {expected_count} prompts")
    return [row["data"] for row in rows]


def preflight_gpu(max_preexisting_bytes=1 << 30):
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is unavailable")
    free, total = torch.cuda.mem_get_info()
    used = total - free
    if used > max_preexisting_bytes:
        raise RuntimeError(
            f"GPU preflight refused benchmark: {used / 2**30:.2f} GiB is already in use "
            f"(limit {max_preexisting_bytes / 2**30:.2f} GiB); no process was terminated")
    return {"free_bytes": free, "total_bytes": total, "preexisting_bytes": used}


def tokenize(tokenizer, text, device="cuda"):
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": text}]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors="pt",
                                            enable_thinking=False)
    else:
        ids = tokenizer(text, add_special_tokens=True, return_tensors="pt").input_ids
    return ids.to(device)


@torch.inference_mode()
def precompile_proposal_shapes(runtime):
    """Materialize every static PARD2 proposal graph outside measured runs.

    After the first round, ``draft_input`` contains the accepted prefix plus
    the target correction/bonus token. Appending ``draft_k - 1`` PARD masks
    therefore gives M in [draft_k, 2 * draft_k], inclusive. Explicitly
    exercising all 16 shapes makes a cold compile cache reproducible instead
    of letting a scored prompt pay an input-dependent Dynamo/Inductor cost.
    """
    if runtime.draft is None:
        return []
    device = next(runtime.draft.parameters()).device
    target_dtype = next(runtime.target.parameters()).dtype
    compiled = []
    k = runtime.spec.draft_k
    for rows in range(k, 2 * k + 1):
        print(f"[precompile] proposal M={rows}", flush=True)
        real_rows = rows - (k - 1)
        input_ids = torch.ones((1, rows), device=device, dtype=torch.long)
        input_ids[:, real_rows:] = runtime.spec.pard_token
        positions = torch.arange(1, rows + 1, device=device)
        cache = runtime._draft_cache()
        kwargs = {}
        if runtime.mode == "pard2-td":
            if runtime.td_unique_projection:
                width = runtime.draft.target_proj.out_features
                kwargs["projected_target_feat"] = torch.zeros(
                    (1, rows, width), device=device,
                    dtype=next(runtime.draft.parameters()).dtype)
            else:
                kwargs["target_feat"] = torch.zeros(
                    (1, rows, runtime.spec.target_dim), device=device,
                    dtype=target_dtype)
        prefix_kwargs = {name: value[:, :1] for name, value in kwargs.items()}
        runtime.draft(
            input_ids=torch.ones((1, 1), device=device, dtype=torch.long),
            past_key_values=cache, cache_position=torch.zeros(
                1, device=device, dtype=torch.long), use_cache=True,
            attention_mask=None, return_dict=True, **prefix_kwargs)
        runtime.draft_forward(
            input_ids=input_ids, past_key_values=cache,
            cache_position=positions, use_cache=True, attention_mask=None,
            return_dict=True, **kwargs)
        torch.cuda.synchronize()
        compiled.append(rows)
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    return compiled


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--mode", required=True, choices=("ar", "pard2-ti", "pard2-td"))
    result.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    result.add_argument("--target", default=str(DEFAULT_TARGET))
    result.add_argument("--draft", default=str(DEFAULT_DRAFT))
    result.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    result.add_argument("--data-root", default=str(DATA_ROOT))
    result.add_argument("--generated-tokens", type=int, default=256)
    result.add_argument("--warmups", type=int, default=8)
    result.add_argument("--sweeps", type=int, default=3)
    result.add_argument("--limit", type=int,
                        help="smoke-only prompt limit; marks output unqualified")
    result.add_argument("--offset", type=int, default=0,
                        help="smoke-only starting prompt index; requires --limit")
    result.add_argument("--compile-mode", default="max-autotune")
    result.add_argument("--precompile-draft-shapes",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="seed all static proposal graphs before timing")
    result.add_argument("--output", required=True)
    result.add_argument("--ignore-eos", action="store_true")
    result.add_argument("--calibration")
    result.add_argument("--quantized-draft")
    result.add_argument("--adaptive-k", action="store_true")
    result.add_argument("--expanded-mha", action="store_true")
    result.add_argument("--fused-decode-append",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="unified prefill/decode/chunk KV4 writer")
    result.add_argument("--exact-row-norm",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="use row-independent HIP RMSNorm for exact chunk parity")
    result.add_argument("--rowwise-lm-head",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="run small-chunk LM-head as independent M=1 launches")
    result.add_argument("--exact-small-chunk",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="legacy alias overriding both independent controls")
    result.add_argument("--td-cache-basis",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="cache TD rotation signs and final norm weight on the target GPU")
    result.add_argument("--td-lazy-features",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="restore only accepted verifier feature rows")
    result.add_argument("--td-unique-projection",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="project real TD rows before expanding PARD mask features")
    result.add_argument("--td-basis-fold",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="fold inverse Hadamard/sign restoration into TD projection")
    result.add_argument("--fused-norm-quant",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="fuse exact layer RMSNorm with activation INT4 packing")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    gpu = preflight_gpu()
    verify_target_checkpoint(args.target)
    if args.mode != "ar": verify_local_resources(args.draft)
    prompts = load_prompts(args.dataset, args.data_root)
    if args.offset and args.limit is None:
        raise ValueError("--offset requires --limit")
    if args.limit is not None:
        if args.limit <= 0 or args.offset < 0:
            raise ValueError("--limit must be positive and --offset non-negative")
        prompts = prompts[args.offset:args.offset + args.limit]
    runtime = load_runtime(mode=args.mode, target_checkpoint=args.target,
        draft_snapshot=args.draft, tokenizer_path=args.tokenizer,
        max_cache_len=8192, compile_mode=args.compile_mode,
        ignore_eos=args.ignore_eos, calibration_path=args.calibration,
        quantized_draft=args.quantized_draft, adaptive_k=args.adaptive_k,
        native_gqa=not args.expanded_mha,
        fused_decode_append=args.fused_decode_append,
        exact_row_norm=args.exact_row_norm,
        rowwise_lm_head=args.rowwise_lm_head,
        exact_small_chunk=args.exact_small_chunk,
        td_cache_basis=args.td_cache_basis,
        td_lazy_features=args.td_lazy_features,
        td_unique_projection=args.td_unique_projection,
        td_basis_fold=args.td_basis_fold,
        fused_norm_quant=args.fused_norm_quant)

    compiled_shapes = []
    if (args.mode != "ar" and args.compile_mode != "eager"
            and args.precompile_draft_shapes):
        if args.compile_mode != "max-autotune-no-cudagraphs":
            raise ValueError("full-shape precompile requires "
                             "max-autotune-no-cudagraphs")
        compiled_shapes = precompile_proposal_shapes(runtime)

    # Warmups are deliberately not selected from the scored 80/80/20 prompts.
    warmup_texts = [f"Warmup {index}: briefly explain integer {index}."
                    for index in range(args.warmups)]
    for text in warmup_texts:
        runtime.generate(tokenize(runtime.tokenizer, text), min(args.generated_tokens, 32))
        gc.collect(); torch.cuda.empty_cache()

    runs = []
    order = list(range(len(prompts)))
    for sweep in range(args.sweeps):
        random.Random(0x50415244 + sweep).shuffle(order)
        for index in order:
            result = runtime.generate(tokenize(runtime.tokenizer, prompts[index]),
                                      args.generated_tokens)
            row = {"sweep": sweep, "prompt_index": index,
                   "output_ids": result.output_ids, **result.metrics()}
            runs.append(row)
            gc.collect(); torch.cuda.empty_cache()
    payload = {
        "contract": {"mode": args.mode, "dataset": args.dataset,
            "qualified": args.limit is None,
            "prompt_count": len(prompts),
            "prompt_offset": args.offset,
            "generated_tokens": args.generated_tokens, "batch_size": 1,
            "warmups": args.warmups, "sweeps": args.sweeps,
            "greedy": True, "ignore_eos": args.ignore_eos,
            "target": args.target, "draft_revision": "67a1516c8f6fc145cda99916799a0cbb3a4af135",
            "upstream_commit": "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038",
            "precompile_draft_shapes": args.precompile_draft_shapes,
            "compiled_proposal_shapes": compiled_shapes,
            "td_cache_basis": args.td_cache_basis,
            "td_lazy_features": args.td_lazy_features,
            "td_unique_projection": args.td_unique_projection,
            "td_basis_fold": args.td_basis_fold,
            "fused_norm_quant": args.fused_norm_quant},
        "gpu_preflight": gpu,
        "runs": runs,
        "median_steady_tokens_per_s": statistics.median(
            row["steady_tokens_per_s"] for row in runs),
        "median_end_to_end_tokens_per_s": statistics.median(
            row["end_to_end_tokens_per_s"] for row in runs),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
