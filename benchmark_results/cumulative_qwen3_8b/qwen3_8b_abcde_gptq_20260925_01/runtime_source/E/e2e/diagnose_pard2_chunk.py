"""Compare a real fused target chunk against sequential AR from the same cache.

This is a correctness diagnostic, not a benchmark. It rebuilds two identical
AR caches, feeds one continuation token at a time to the first cache and the
same continuation as one chunk to the second cache, then reports the first
transformer stage whose outputs differ.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.benchmark_pard2 import load_prompts, tokenize
from e2e.pard2 import DEFAULT_TARGET, DEFAULT_TOKENIZER
from e2e.speculative import load_runtime


def _tensor(output):
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"hook output is not a tensor: {type(output)!r}")
    return output.detach().clone()


class Capture:
    def __init__(self, target):
        self.values = defaultdict(list)
        self.handles = []
        for index, layer in enumerate(target.model.layers):
            for stage, module in (
                ("attention", layer.self_attn),
                ("mlp", layer.mlp),
                ("layer", layer),
            ):
                name = f"layer_{index:02d}.{stage}"
                self.handles.append(module.register_forward_hook(self._hook(name)))
        self.handles.append(
            target.model.norm.register_forward_hook(self._hook("final_norm")))
        self.handles.append(
            target.lm_head.register_forward_hook(self._hook("lm_head")))

    def _hook(self, name):
        def save(_module, _inputs, output):
            self.values[name].append(_tensor(output))
        return save

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _call(runtime, ids, cache):
    positions = torch.arange(cache.length, cache.length + ids.shape[1],
                             device=ids.device)
    output, _ = runtime._target_call(ids, cache, positions)
    return output


def _build_ar_cache(runtime, prompt_ids, prefix):
    cache = runtime._target_cache()
    _call(runtime, prompt_ids, cache)
    for token in prefix:
        ids = torch.tensor([[token]], device=prompt_ids.device,
                           dtype=prompt_ids.dtype)
        _call(runtime, ids, cache)
    return cache


def _logical_cache_differences(left, right):
    full_pages, remainder = divmod(left.length, left.page_size)
    data, scales = [], []
    for layer in range(left.n_layers):
        data_diff = scale_diff = 0
        for page in range(full_pages + bool(remainder)):
            kept = left.page_size
            if page == full_pages:
                kept = remainder
            data_diff += int(left.pages[page, layer, :, :, :kept].ne(
                right.pages[page, layer, :, :, :kept]).sum().item())
            scale_diff += int(left.scales[page, layer, :, :, :kept].ne(
                right.scales[page, layer, :, :, :kept]).sum().item())
        data.append(data_diff)
        scales.append(scale_diff)
    return data, scales


def _joined(values):
    return torch.cat(values, dim=1) if len(values) > 1 else values[0]


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="math_500")
    parser.add_argument("--prompt-index", type=int, default=14)
    parser.add_argument("--start", type=int, default=160)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--ar-result", default="pard2_formal_results/ar_math_500.json")
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--expanded-mha", action="store_true")
    parser.add_argument("--fused-decode-append",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.ar_result).read_text())
    rows = [row for row in payload["runs"]
            if row["prompt_index"] == args.prompt_index and row["sweep"] == 0]
    if len(rows) != 1:
        raise ValueError("AR result must contain exactly one requested prompt/sweep")
    generated = rows[0]["output_ids"]
    end = args.start + args.chunk
    if not 0 <= args.start < end <= len(generated):
        raise ValueError("requested continuation is outside the AR result")

    runtime = load_runtime(
        mode="ar", target_checkpoint=args.target, draft_snapshot="",
        tokenizer_path=args.tokenizer, max_cache_len=8192,
        compile_mode="eager", native_gqa=not args.expanded_mha,
        fused_decode_append=args.fused_decode_append,
    )
    prompt = load_prompts(args.dataset)[args.prompt_index]
    prompt_ids = tokenize(runtime.tokenizer, prompt)
    prefix = generated[:args.start]
    continuation = generated[args.start:end]

    sequential_cache = _build_ar_cache(runtime, prompt_ids, prefix)
    chunk_cache = _build_ar_cache(runtime, prompt_ids, prefix)
    if sequential_cache.length != chunk_cache.length:
        raise RuntimeError("base cache lengths differ")

    sequential_capture = Capture(runtime.target)
    sequential_logits = []
    for token in continuation:
        ids = torch.tensor([[token]], device=prompt_ids.device,
                           dtype=prompt_ids.dtype)
        sequential_logits.append(_call(runtime, ids, sequential_cache).logits)
    sequential_capture.close()
    sequential_logits = torch.cat(sequential_logits, dim=1)

    chunk_capture = Capture(runtime.target)
    chunk_ids = torch.tensor([continuation], device=prompt_ids.device,
                             dtype=prompt_ids.dtype)
    chunk_logits = _call(runtime, chunk_ids, chunk_cache).logits
    chunk_capture.close()
    torch.cuda.synchronize()
    cache_data_diff, cache_scale_diff = _logical_cache_differences(
        sequential_cache, chunk_cache)

    diagnostics = []
    first_different_stage = None
    for name in sequential_capture.values:
        sequential = _joined(sequential_capture.values[name])
        chunked = _joined(chunk_capture.values[name])
        delta = (sequential.float() - chunked.float()).abs()
        item = {
            "stage": name,
            "exact": bool(torch.equal(sequential, chunked)),
            "max_abs": float(delta.max()),
            "mean_abs": float(delta.mean()),
            "nonzero": int(torch.count_nonzero(delta)),
        }
        diagnostics.append(item)
        if not item["exact"] and first_different_stage is None:
            first_different_stage = name

    top1_sequential = sequential_logits.argmax(-1)
    top1_chunk = chunk_logits.argmax(-1)
    mismatched_positions = torch.nonzero(
        top1_sequential.ne(top1_chunk), as_tuple=False)[:, 1].tolist()
    logit_delta = (sequential_logits.float() - chunk_logits.float()).abs()
    report = {
        "dataset": args.dataset,
        "prompt_index": args.prompt_index,
        "start": args.start,
        "chunk": args.chunk,
        "native_gqa": not args.expanded_mha,
        "fused_decode_append": args.fused_decode_append,
        "base_cache_length": chunk_cache.length - args.chunk,
        "logical_cache_data_exact": not any(cache_data_diff),
        "logical_cache_scale_exact": not any(cache_scale_diff),
        "logical_cache_data_differences_by_layer": cache_data_diff,
        "logical_cache_scale_differences_by_layer": cache_scale_diff,
        "first_different_stage": first_different_stage,
        "logits_exact": bool(torch.equal(sequential_logits, chunk_logits)),
        "logits_max_abs": float(logit_delta.max()),
        "logits_mean_abs": float(logit_delta.mean()),
        "top1_equal": not mismatched_positions,
        "top1_mismatched_positions": mismatched_positions,
        "sequential_top1": top1_sequential[0].tolist(),
        "chunk_top1": top1_chunk[0].tolist(),
        "stages": diagnostics,
    }
    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n")


if __name__ == "__main__":
    main()
