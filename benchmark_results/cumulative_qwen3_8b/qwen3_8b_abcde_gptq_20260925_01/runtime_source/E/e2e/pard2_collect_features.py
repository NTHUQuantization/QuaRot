"""Collect Qwen3 BF16/fused feature pairs from the frozen PARD2 tune split."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2e.benchmark_pard2 import DATA_ROOT, DATASETS, load_prompts, preflight_gpu, tokenize
from e2e.pard2 import DEFAULT_DRAFT, DEFAULT_TARGET, DEFAULT_TOKENIZER
from e2e.pard2_calibrate import fit_affine
from e2e.speculative import (
    Pard2Spec, SelectedHiddenCollector, load_td_target_basis)


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--source", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--draft", default=str(DEFAULT_DRAFT))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--prompts-per-dataset", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.prompts_per_dataset <= 4:
        raise ValueError("the frozen tune split contains 1..4 prompts per dataset")
    gpu = preflight_gpu()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from e2e.model_registry import runtime_types

    tokenizer = AutoTokenizer.from_pretrained(args.source, local_files_only=True)
    reference = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager").eval().to("cuda")
    config_cls, target_cls, _ = runtime_types(args.target, local_files_only=True)
    target_config = config_cls.from_pretrained(
        args.target, local_files_only=True, attn_implementation="flash_attention_2")
    target = target_cls.from_pretrained(
        args.target, config=target_config, torch_dtype=torch.float16,
        local_files_only=True).eval().to("cuda")
    signs, final_norm = load_td_target_basis(target, args.source)
    collector = SelectedHiddenCollector(
        target, Pard2Spec().target_layers, signs, final_norm)

    fused_parts, reference_parts, manifest = [], [], []
    for dataset in DATASETS:
        prompts = load_prompts(dataset, args.data_root)[:args.prompts_per_dataset]
        for index, prompt in enumerate(prompts):
            ids = tokenize(tokenizer, prompt)
            reference_output = reference(
                input_ids=ids, use_cache=False, output_hidden_states=True,
                return_dict=True, attention_mask=None)
            reference_feature = torch.cat(
                [reference_output.hidden_states[layer]
                 for layer in Pard2Spec().target_layers], dim=-1)
            collector.reset()
            target(input_ids=ids, use_cache=False, output_hidden_states=False,
                   return_dict=True, attention_mask=None)
            fused_feature = collector.features()
            if fused_feature.shape != reference_feature.shape:
                raise RuntimeError("reference/fused feature shapes differ")
            fused_parts.append(fused_feature.cpu().half())
            reference_parts.append(reference_feature.cpu().half())
            manifest.append({"dataset": dataset, "index": index,
                             "tokens": int(ids.shape[1]), "split": "tune"})
            del reference_output, reference_feature, fused_feature
            torch.cuda.empty_cache()

    collector.close()
    fused_raw = torch.cat(fused_parts, dim=1)
    reference_raw = torch.cat(reference_parts, dim=1)
    raw_scale, raw_bias = fit_affine(fused_raw, reference_raw)

    state = torch.load(
        Path(args.draft) / "warp_model.bin", map_location="cpu", weights_only=True)
    weight = state["target_proj.weight"].to("cuda", dtype=torch.bfloat16)
    bias = state.get("target_proj.bias")
    if bias is not None:
        bias = bias.to("cuda", dtype=torch.bfloat16)
    fused_projected, reference_projected = [], []
    for fused_part, reference_part in zip(fused_parts, reference_parts):
        calibrated = (
            fused_part.to("cuda", dtype=torch.bfloat16)
            * raw_scale.to("cuda", dtype=torch.bfloat16)
            + raw_bias.to("cuda", dtype=torch.bfloat16))
        fused_projected.append(F.linear(calibrated, weight, bias).cpu().half())
        reference_projected.append(F.linear(
            reference_part.to("cuda", dtype=torch.bfloat16), weight, bias).cpu().half())

    del reference, target, weight, bias
    gc.collect()
    torch.cuda.empty_cache()
    payload = {
        "split": "tune",
        "runtime": "fused_v1",
        "target_model": Pard2Spec().target_model_id,
        "target_revision": Pard2Spec().target_revision,
        "draft_revision": Pard2Spec().revision,
        "dataset_hashes": {name: digest for name, (_, digest) in DATASETS.items()},
        "gpu_preflight": gpu,
        "manifest": manifest,
        "fused_raw": fused_raw,
        "reference_raw": reference_raw,
        "fused_projected": torch.cat(fused_projected, dim=1),
        "reference_projected": torch.cat(reference_projected, dim=1),
    }
    torch.save(payload, Path(args.output))
    print({"output": args.output, "tokens": int(fused_raw.shape[1]),
           "raw_dim": int(fused_raw.shape[-1]),
           "projected_dim": int(payload["fused_projected"].shape[-1])})


if __name__ == "__main__":
    main()
