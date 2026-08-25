"""CLI for fused_v1 AR, PARD2-TI and PARD2-TD generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.speculative import Pard2Spec, load_runtime


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET = ROOT / "qwen3_8b_fused_v1_rtn_w4a4kv4"
DEFAULT_DRAFT = ROOT / ".hf_cache/pard/hub/models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135"
DEFAULT_TOKENIZER = ROOT / ".hf_cache/pard/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"


def verify_local_resources(draft=DEFAULT_DRAFT):
    required = {
        "config.json": 1,
        "model.safetensors": 1_000_000_000,
        "warp_model.bin": 1_000_000,
    }
    failures = []
    for name, minimum in required.items():
        path = Path(draft) / name
        if not path.is_file() or path.stat().st_size < minimum:
            failures.append(str(path))
    if failures:
        spec = Pard2Spec()
        raise FileNotFoundError(
            "pinned PARD2-Qwen3 cache is incomplete; run only this missing-file download:\n"
            f"hf download {spec.model_id} --revision {spec.revision}\n"
            + "\n".join(failures))


def verify_target_checkpoint(target):
    path = Path(target)
    if not (path / "config.json").is_file():
        spec = Pard2Spec()
        raise FileNotFoundError(
            f"missing fused Qwen3-8B W4A4KV4 target: {path}\n"
            "Create it from the locally pinned BF16 source "
            f"{spec.target_model_id}@{spec.target_revision} with "
            "e2e/checkpoint_utils/quantize_checkpoint.py.")


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("ar", "pard2-ti", "pard2-td"), required=True)
    result.add_argument("--target", default=str(DEFAULT_TARGET))
    result.add_argument("--draft", default=str(DEFAULT_DRAFT))
    result.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    result.add_argument("--prompt", default="Explain speculative decoding in one paragraph.")
    result.add_argument("--max-new-tokens", type=int, default=256)
    result.add_argument("--max-cache-len", type=int, default=4096)
    result.add_argument("--page-size", type=int, default=128)
    result.add_argument("--compile-mode", default="max-autotune",
                        choices=("eager", "default", "reduce-overhead", "max-autotune"))
    result.add_argument("--ignore-eos", action="store_true")
    result.add_argument("--calibration")
    result.add_argument("--quantized-draft")
    result.add_argument("--adaptive-k", action="store_true")
    result.add_argument("--expanded-mha", action="store_true",
                        help="correctness oracle: expand native KV heads")
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
    result.add_argument("--json-output")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    verify_target_checkpoint(args.target)
    if args.mode != "ar":
        verify_local_resources(args.draft)
    runtime = load_runtime(mode=args.mode, target_checkpoint=args.target,
        draft_snapshot=args.draft, tokenizer_path=args.tokenizer,
        max_cache_len=args.max_cache_len, page_size=args.page_size,
        compile_mode=args.compile_mode, ignore_eos=args.ignore_eos,
        calibration_path=args.calibration, quantized_draft=args.quantized_draft,
        adaptive_k=args.adaptive_k, native_gqa=not args.expanded_mha,
        fused_decode_append=args.fused_decode_append,
        exact_row_norm=args.exact_row_norm,
        rowwise_lm_head=args.rowwise_lm_head,
        exact_small_chunk=args.exact_small_chunk,
        td_cache_basis=args.td_cache_basis,
        td_lazy_features=args.td_lazy_features,
        td_unique_projection=args.td_unique_projection,
        td_basis_fold=args.td_basis_fold,
        fused_norm_quant=args.fused_norm_quant)
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt}]
    if getattr(runtime.tokenizer, "chat_template", None):
        ids = runtime.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False)
    else:
        ids = runtime.tokenizer(args.prompt, return_tensors="pt").input_ids
    result = runtime.generate(ids.to("cuda"), args.max_new_tokens)
    payload = result.metrics()
    payload["mode"] = args.mode
    payload["text"] = runtime.tokenizer.decode(result.output_ids, skip_special_tokens=False)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if args.json_output:
        Path(args.json_output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
