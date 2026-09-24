#!/usr/bin/env python3
"""Compare streamed BF16 and QuaRot Qwen3-32B next-token distributions."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RotaryEmbedding)

from e2e.benchmark_pard2 import (
    DATA_ROOT, DATASETS, _hip_extension_provenance, load_prompts, sha256,
    target_profile_preflight, token_ids_sha256, tokenize)


class SafeTensorSource:
    def __init__(self, root):
        self.root = Path(root)
        index = json.loads(
            (self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]

    def tensors(self, keys):
        grouped = {}
        for key in keys:
            grouped.setdefault(self.weight_map[key], []).append(key)
        result = {}
        for shard, shard_keys in grouped.items():
            with safe_open(
                    self.root / shard, framework="pt", device="cpu") as handle:
                for key in shard_keys:
                    result[key] = handle.get_tensor(key)
        return result

    def tensor(self, key):
        return self.tensors([key])[key]


def _gpu_preflight():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm device is unavailable")
    free, total = torch.cuda.mem_get_info()
    used = total - free
    if used > 1 << 30:
        raise RuntimeError(
            f"GPU preflight refused: {used / 2**30:.2f} GiB already in use")
    return {"free_bytes": free, "total_bytes": total,
            "preexisting_bytes": used}


def _prompt_records(args, tokenizer, device):
    records = []
    for dataset in args.datasets:
        prompts = load_prompts(dataset, args.data_root)
        for offset, prompt in enumerate(prompts[:args.limit_per_dataset]):
            ids = tokenize(tokenizer, prompt, device=device)
            records.append({
                "dataset": dataset,
                "offset": offset,
                "input_ids_sha256": token_ids_sha256(ids),
                "input_token_count": int(ids.numel()),
                "input_ids": ids,
            })
    return records


def _new_layer(config, index, dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        return Qwen3DecoderLayer(config, index)
    finally:
        torch.set_default_dtype(previous)


def _decoder_hidden(output):
    if torch.is_tensor(output):
        return output
    return output[0]


@torch.inference_mode()
def write_reference(args):
    gpu = _gpu_preflight()
    source = SafeTensorSource(args.source)
    config = AutoConfig.from_pretrained(
        args.source, local_files_only=True,
        attn_implementation="flash_attention_2")
    config._attn_implementation = "flash_attention_2"
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    records = _prompt_records(args, tokenizer, device)

    embedding = source.tensor("model.embed_tokens.weight").to(
        device=device, dtype=dtype)
    hidden_states = [
        F.embedding(record["input_ids"], embedding) for record in records]
    del embedding
    torch.cuda.empty_cache()

    rotary = Qwen3RotaryEmbedding(config=config).to(device=device)
    positions = []
    position_embeddings = []
    for hidden in hidden_states:
        cache_position = torch.arange(hidden.shape[1], device=device)
        position_ids = cache_position.unsqueeze(0)
        positions.append((cache_position, position_ids))
        position_embeddings.append(rotary(hidden, position_ids))
    del rotary

    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}."
        keys = [
            key for key in source.weight_map if key.startswith(prefix)]
        tensors = source.tensors(keys)
        state = {key[len(prefix):]: value for key, value in tensors.items()}
        state.pop("self_attn.rotary_emb.inv_freq", None)
        layer = _new_layer(config, index, dtype)
        result = layer.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"reference layer load failed: {result}")
        del state, tensors
        layer = layer.eval().to(device)
        for sample, hidden in enumerate(hidden_states):
            cache_position, position_ids = positions[sample]
            hidden_states[sample] = _decoder_hidden(layer(
                hidden, attention_mask=None, position_ids=position_ids,
                position_embeddings=position_embeddings[sample],
                cache_position=cache_position, use_cache=False))
        del layer
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"BF16 reference layer {index + 1}/{config.num_hidden_layers}",
            flush=True)

    norm = source.tensor("model.norm.weight").to(device=device, dtype=dtype)
    head = source.tensor("lm_head.weight").to(device=device, dtype=dtype)
    logits = []
    for hidden in hidden_states:
        final = hidden[:, -1:].float()
        variance = final.square().mean(dim=-1, keepdim=True)
        final = (
            final * torch.rsqrt(variance + config.rms_norm_eps)).to(dtype)
        final = final * norm
        logits.append(F.linear(final[:, -1], head).float().cpu())
    source_root = Path(args.source).resolve()
    payload = {
        "kind": "bf16_reference",
        "source": str(source_root),
        "source_provenance": {
            "config_sha256": sha256(source_root / "config.json"),
            "index_sha256": sha256(
                source_root / "model.safetensors.index.json"),
        },
        "gpu_preflight": gpu,
        "records": [{
            key: value for key, value in record.items() if key != "input_ids"
        } for record in records],
        "logits": logits,
    }
    torch.save(payload, args.output)
    print(json.dumps({
        "kind": payload["kind"], "records": len(records),
        "output": str(args.output),
    }, indent=2))


@torch.inference_mode()
def write_target(args):
    target_provenance = target_profile_preflight(
        args.checkpoint, "qwen3_32b")
    gpu = _gpu_preflight()
    from e2e.speculative import load_runtime

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    records = _prompt_records(args, tokenizer, torch.device("cuda:0"))
    runtime = load_runtime(
        mode="ar", target_checkpoint=args.checkpoint,
        draft_snapshot=None, tokenizer_path=args.tokenizer,
        max_cache_len=args.max_cache_len, compile_mode="eager",
        ignore_eos=False, native_gqa=True, fused_decode_append=True,
        exact_row_norm=True, rowwise_lm_head=False)
    try:
        hip_extension = _hip_extension_provenance(
            args.expected_hip_sha256)
        logits = []
        for record in records:
            ids = record["input_ids"]
            cache = runtime._target_cache()
            positions = torch.arange(ids.shape[1], device=ids.device)
            output, _ = runtime._target_call(ids, cache, positions)
            logits.append(output.logits[:, -1].float().cpu())
            del cache, output
    finally:
        runtime.close()
    config = json.loads((Path(args.checkpoint) / "config.json").read_text())
    payload = {
        "kind": "quarot_target",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "conversion": config["quarot_conversion"],
        "target_provenance": target_provenance,
        "hip_extension": hip_extension,
        "gpu_preflight": gpu,
        "records": [{
            key: value for key, value in record.items() if key != "input_ids"
        } for record in records],
        "logits": logits,
    }
    torch.save(payload, args.output)
    print(json.dumps({
        "kind": payload["kind"],
        "method": payload["conversion"]["method"],
        "records": len(records), "output": str(args.output),
    }, indent=2))


def _metrics(reference, candidate):
    reference = reference.double()
    candidate = candidate.double()
    reference_logp = F.log_softmax(reference, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    reference_p = reference_logp.exp()
    candidate_p = candidate_logp.exp()
    difference = candidate - reference
    topk = min(5, reference.numel(), candidate.numel())
    reference_top5 = set(reference.topk(topk).indices.tolist())
    candidate_top5 = set(candidate.topk(topk).indices.tolist())
    return {
        "logit_rmse": math.sqrt(float(difference.square().mean())),
        "logit_mae": float(difference.abs().mean()),
        "logit_cosine": float(F.cosine_similarity(
            reference.unsqueeze(0), candidate.unsqueeze(0))),
        "kl_reference_to_candidate": float(
            (reference_p * (reference_logp - candidate_logp)).sum()),
        "total_variation": float(
            0.5 * (reference_p - candidate_p).abs().sum()),
        "top1_match": bool(reference.argmax() == candidate.argmax()),
        "top5_overlap": len(reference_top5 & candidate_top5),
        "reference_top1_token": int(reference.argmax()),
        "candidate_top1_token": int(candidate.argmax()),
    }


def compare(args):
    reference = torch.load(
        args.reference, map_location="cpu", weights_only=True)
    candidates = {
        "rtn": torch.load(args.rtn, map_location="cpu", weights_only=True),
        "gptq": torch.load(args.gptq, map_location="cpu", weights_only=True),
    }
    for method, candidate in candidates.items():
        actual_method = candidate.get("conversion", {}).get("method")
        if actual_method != method:
            raise ValueError(
                f"{method} candidate reports conversion method {actual_method!r}")
    rows = []
    for name, candidate in candidates.items():
        if candidate["records"] != reference["records"]:
            raise ValueError(f"{name} prompt records differ from reference")
        for record, ref_logits, candidate_logits in zip(
                reference["records"], reference["logits"],
                candidate["logits"], strict=True):
            rows.append({
                "method": name, **record,
                **_metrics(ref_logits[0], candidate_logits[0]),
            })
    aggregates = {}
    for name in candidates:
        selected = [row for row in rows if row["method"] == name]
        numeric = (
            "logit_rmse", "logit_mae", "logit_cosine",
            "kl_reference_to_candidate", "total_variation",
            "top5_overlap")
        aggregates[name] = {
            key: sum(row[key] for row in selected) / len(selected)
            for key in numeric}
        aggregates[name]["top1_match_rate"] = (
            sum(row["top1_match"] for row in selected) / len(selected))
    verdict = {
        metric: aggregates["gptq"][metric] < aggregates["rtn"][metric]
        for metric in (
            "logit_rmse", "logit_mae",
            "kl_reference_to_candidate", "total_variation")}
    result = {
        "status": "passed", "reference": str(args.reference),
        "candidates": {"rtn": str(args.rtn), "gptq": str(args.gptq)},
        "records": rows, "aggregates": aggregates,
        "gptq_lower_gap_than_rtn": verdict,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tokenizer", required=True)
    common.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS),
        default=list(DATASETS))
    common.add_argument("--data-root", type=Path, default=DATA_ROOT)
    common.add_argument("--limit-per-dataset", type=int, default=1)
    common.add_argument("--output", type=Path, required=True)

    reference = subparsers.add_parser("reference", parents=[common])
    reference.add_argument("--source", required=True)

    target = subparsers.add_parser("target", parents=[common])
    target.add_argument("--checkpoint", required=True)
    target.add_argument("--max-cache-len", type=int, default=2048)
    target.add_argument(
        "--expected-hip-sha256", required=True)

    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--rtn", type=Path, required=True)
    comparison.add_argument("--gptq", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.mode in ("reference", "target"):
        if args.limit_per_dataset <= 0:
            raise ValueError("--limit-per-dataset must be positive")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "reference":
        write_reference(args)
    elif args.mode == "target":
        write_target(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
