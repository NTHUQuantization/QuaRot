"""Extract resumable GPTQ-32B teacher shards for projection-only PARD-2 training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch

from e2e.qwen3_32b_td_features import (
    atomic_json,
    gpu_snapshot,
    load_quantized_target,
    preflight_gpu,
    sha256_file,
)
from e2e.qwen3_32b_td_numeric import install_safe_collector


TAPS = (-1, -8, -16, -24)
MIN_DISK_RESERVE = 25 * 1024**3


def iter_stage_rows(path, start_tokens, end_tokens):
    cumulative = 0
    with Path(path).open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            row = json.loads(line)
            next_cumulative = cumulative + int(row["token_count"])
            if next_cumulative <= start_tokens:
                cumulative = next_cumulative
                continue
            if cumulative >= end_tokens:
                break
            yield ordinal, row, cumulative, next_cumulative
            cumulative = next_cumulative


def atomic_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(payload, tmp)
        if tmp.stat().st_size + MIN_DISK_RESERVE > tmp.stat().st_dev * 0 + _free_bytes(path.parent):
            raise RuntimeError("teacher shard would violate 25 GiB disk reserve")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _free_bytes(path):
    import os
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def target_forward(target, collector, ids):
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    collector.reset()
    with torch.inference_mode():
        outputs = target(
            input_ids=input_ids,
            attention_mask=None,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        features = collector.features().detach()
        logits = outputs.logits.float()
        gold_prob = torch.ones((1, len(ids)), device="cuda", dtype=torch.float32)
        if len(ids) > 1:
            labels = input_ids[:, 1:].unsqueeze(-1)
            gold_prob[:, 1:] = torch.log_softmax(logits[:, :-1], -1).gather(-1, labels).squeeze(-1).exp()
    if tuple(features.shape) != (1, len(ids), 20480):
        raise RuntimeError(f"teacher feature shape mismatch: {features.shape}")
    if not torch.isfinite(features).all() or not torch.isfinite(gold_prob).all():
        raise RuntimeError("non-finite teacher cache tensor")
    return features.cpu().to(torch.bfloat16), gold_prob.cpu()


def flush(output_dir, index, rows, features, probs):
    feature = torch.cat(features, dim=1).contiguous()
    probability = torch.cat(probs, dim=1).contiguous()
    if feature.shape[:2] != probability.shape:
        raise RuntimeError("teacher feature/probability token mismatch")
    path = output_dir / f"teacher-{index:04d}.pt"
    atomic_save(path, {"schema_version": 1, "taps": TAPS, "rows": rows,
                       "features": feature, "teacher_gold_prob": probability})
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "bytes": path.stat().st_size, "tokens": int(feature.shape[1]),
            "samples": len(rows), "shape": list(feature.shape)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--start-tokens", type=int, required=True)
    parser.add_argument("--end-tokens", type=int, required=True)
    parser.add_argument("--shard-tokens", type=int, default=16000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.start_tokens < args.end_tokens:
        raise ValueError("invalid cumulative token range")
    if _free_bytes(args.output_dir.parent) < MIN_DISK_RESERVE + args.shard_tokens * 41000:
        raise RuntimeError("insufficient disk for next teacher shard plus reserve")

    install_safe_collector()
    from e2e.speculative import SelectedHiddenCollector, load_td_target_basis
    preflight = preflight_gpu()
    target = load_quantized_target(args.target)
    signs, final_norm = load_td_target_basis(target, args.source)
    collector = SelectedHiddenCollector(target, TAPS, signs, final_norm, cache_basis=True)
    snapshots = [gpu_snapshot("runtime_loaded")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, features, probs, shard_tokens, shards = [], [], [], 0, []
    started = time.perf_counter(); processed = 0
    first_cumulative = last_cumulative = None
    for sample_index, (ordinal, row, cumulative, next_cumulative) in enumerate(
        iter_stage_rows(args.data, args.start_tokens, args.end_tokens)
    ):
        ids = [int(x) for x in row["input_ids"]]
        feature, prob = target_forward(target, collector, ids)
        rows.append({"ordinal": ordinal, "sample_id": row["sample_id"],
                     "token_ids_sha256": row["token_ids_sha256"],
                     "tokens": len(ids), "input_ids": ids, "loss_start": int(row["loss_start"]),
                     "stratum": row["stratum"]})
        features.append(feature); probs.append(prob)
        shard_tokens += len(ids); processed += len(ids)
        first_cumulative = cumulative if first_cumulative is None else first_cumulative
        last_cumulative = next_cumulative
        if shard_tokens >= args.shard_tokens:
            shards.append(flush(args.output_dir, len(shards), rows, features, probs))
            rows, features, probs, shard_tokens = [], [], [], 0
            gc.collect(); torch.cuda.empty_cache()
            snapshots.append(gpu_snapshot(f"shard_{len(shards)}"))
    if features:
        shards.append(flush(args.output_dir, len(shards), rows, features, probs))
    elapsed = time.perf_counter() - started
    snapshots.append(gpu_snapshot("complete")); collector.close()
    manifest = {
        "schema_version": 1, "stage": "TD32-PROJECTION-TEACHER-CACHE",
        "target": str(Path(args.target).resolve()), "source": str(Path(args.source).resolve()),
        "data": str(args.data.resolve()), "data_sha256": sha256_file(args.data),
        "taps": TAPS, "requested_start_tokens": args.start_tokens,
        "requested_end_tokens": args.end_tokens, "actual_start_tokens": first_cumulative,
        "actual_end_tokens": last_cumulative, "processed_tokens": processed,
        "elapsed_seconds": elapsed, "tokens_per_second": processed / elapsed,
        "shards": shards, "disk_free_bytes": _free_bytes(args.output_dir),
        "gpu_preflight": preflight, "memory_snapshots": snapshots,
    }
    atomic_json(args.output_dir / "teacher_manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"memory_snapshots", "shards"}}, indent=2))


if __name__ == "__main__":
    main()
