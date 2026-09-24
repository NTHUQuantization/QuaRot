"""S1/S2 selected-feature extraction and streamed affine fitting for TD32."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import torch


TAP_CANDIDATES = {
    "legacy": (-1, -8, -16, -24),
    "relative": (-1, -13, -26, -38),
    "quartile": (-1, -16, -32, -48),
}
REFERENCE_TAPS = (-1, -8, -16, -24)
UNION_TAPS = tuple(dict.fromkeys(tap for taps in TAP_CANDIDATES.values() for tap in taps))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def atomic_torch_save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def selected_samples(path: Path, token_budget: int):
    remaining = token_budget
    for row in read_jsonl(path):
        ids = [int(value) for value in row["input_ids"]]
        if remaining <= 0:
            break
        ids = ids[:remaining]
        if len(ids) < 8:
            break
        yield row, ids
        remaining -= len(ids)
    if remaining > 0:
        raise RuntimeError(f"data file is {remaining} tokens short of requested budget")


def gpu_snapshot(stage):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    used = total - free
    payload = {
        "stage": stage,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "global_used_bytes": int(used),
        "device_total_bytes": int(total),
        "global_used_percent": 100.0 * used / total,
    }
    if payload["global_used_percent"] >= 95.0:
        raise RuntimeError(f"95% VRAM gate hit at {stage}: {payload}")
    return payload


def preflight_gpu():
    free, total = torch.cuda.mem_get_info()
    used = total - free
    if used > 1024**3:
        raise RuntimeError(f"GPU is not idle before TD32 stage: {used / 2**20:.1f} MiB")
    return {"device_total_bytes": int(total), "device_used_bytes": int(used)}


def load_reference(model_path: str):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    ).eval().to("cuda")


def load_quantized_target(model_path: str):
    from e2e.model_registry import runtime_types

    config_cls, model_cls, _ = runtime_types(model_path, local_files_only=True)
    config = config_cls.from_pretrained(
        model_path,
        local_files_only=True,
        attn_implementation="flash_attention_2",
    )
    target = model_cls.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).eval().to("cuda")
    return target


def model_forward(model, ids):
    input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        model.model(
            input_ids=input_ids,
            attention_mask=None,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )


class SufficientStats:
    def __init__(self, width):
        self.count = 0
        self.sum_x = torch.zeros(width, dtype=torch.float64)
        self.sum_y = torch.zeros(width, dtype=torch.float64)
        self.sum_x2 = torch.zeros(width, dtype=torch.float64)
        self.sum_y2 = torch.zeros(width, dtype=torch.float64)
        self.sum_xy = torch.zeros(width, dtype=torch.float64)

    def update(self, source, reference):
        if source.shape != reference.shape or source.ndim != 2:
            raise ValueError("paired features must have equal [tokens, channels] shape")
        if not torch.isfinite(source).all() or not torch.isfinite(reference).all():
            raise RuntimeError("non-finite paired feature")
        x, y = source.float(), reference.float()
        self.count += int(x.shape[0])
        self.sum_x += x.sum(0).double().cpu()
        self.sum_y += y.sum(0).double().cpu()
        self.sum_x2 += x.square().sum(0).double().cpu()
        self.sum_y2 += y.square().sum(0).double().cpu()
        self.sum_xy += (x * y).sum(0).double().cpu()

    def fit(self):
        if not self.count:
            raise RuntimeError("cannot fit empty feature statistics")
        n = float(self.count)
        mean_x, mean_y = self.sum_x / n, self.sum_y / n
        variance_x = torch.clamp(self.sum_x2 / n - mean_x.square(), min=0)
        variance_y = torch.clamp(self.sum_y2 / n - mean_y.square(), min=0)
        covariance = self.sum_xy / n - mean_x * mean_y
        scale = torch.where(
            variance_x > 1e-12, covariance / variance_x, torch.ones_like(variance_x)
        )
        bias = mean_y - scale * mean_x
        before_mse = torch.clamp(
            (self.sum_x2 + self.sum_y2 - 2 * self.sum_xy).sum() / n, min=0
        ) / scale.numel()
        residual_var = torch.clamp(
            variance_y - torch.where(
                variance_x > 1e-12,
                covariance.square() / variance_x,
                torch.zeros_like(variance_x),
            ),
            min=0,
        )
        after_mse = residual_var.mean()
        before_cos = self.sum_xy.sum() / torch.sqrt(
            self.sum_x2.sum() * self.sum_y2.sum()
        )
        sum_z = scale * self.sum_x + bias * n
        sum_z2 = (
            scale.square() * self.sum_x2
            + 2 * scale * bias * self.sum_x
            + bias.square() * n
        )
        sum_zy = scale * self.sum_xy + bias * self.sum_y
        after_cos = sum_zy.sum() / torch.sqrt(sum_z2.sum() * self.sum_y2.sum())
        metrics = {
            "tokens": self.count,
            "channels": int(scale.numel()),
            "rmse_before": math.sqrt(float(before_mse)),
            "rmse_after": math.sqrt(float(after_mse)),
            "cosine_before": float(before_cos),
            "cosine_after": float(after_cos),
            "scale_min": float(scale.min()),
            "scale_max": float(scale.max()),
            "bias_min": float(bias.min()),
            "bias_max": float(bias.max()),
            "zero_variance_channels": int((variance_x <= 1e-12).sum()),
        }
        if not all(math.isfinite(value) for value in metrics.values() if isinstance(value, float)):
            raise RuntimeError(f"non-finite affine metrics: {metrics}")
        return scale.float(), bias.float(), metrics


def run_s1(args):
    from e2e.speculative import SelectedHiddenCollector, load_td_target_basis

    preflight = preflight_gpu()
    target = load_quantized_target(args.target)
    signs, final_norm = load_td_target_basis(target, args.source)
    collector = SelectedHiddenCollector(
        target, UNION_TAPS, signs, final_norm, cache_basis=True
    )
    snapshots = [gpu_snapshot("runtime_loaded")]
    sums = {
        tap: {
            "count": 0,
            "sum": 0.0,
            "sum2": 0.0,
            "min": math.inf,
            "max": -math.inf,
        }
        for tap in UNION_TAPS
    }
    deterministic = []
    started = time.perf_counter()
    processed = 0
    for sample_index, (row, ids) in enumerate(selected_samples(args.data, args.token_budget)):
        collector.reset()
        model_forward(target, ids)
        features = collector.features().detach()
        if features.shape != (1, len(ids), len(UNION_TAPS) * 5120):
            raise RuntimeError(f"unexpected union feature shape {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise RuntimeError("S1 feature contains NaN or Inf")
        chunks = features.split(5120, dim=-1)
        for tap, chunk in zip(UNION_TAPS, chunks):
            value = chunk.float()
            stat = sums[tap]
            stat["count"] += value.numel()
            stat["sum"] += float(value.sum())
            stat["sum2"] += float(value.square().sum())
            stat["min"] = min(stat["min"], float(value.min()))
            stat["max"] = max(stat["max"], float(value.max()))
        if sample_index < args.determinism_samples:
            first = features.cpu()
            collector.reset()
            model_forward(target, ids)
            second = collector.features().detach().cpu()
            difference = (first.float() - second.float()).abs()
            deterministic.append(
                {
                    "sample_id": row["sample_id"],
                    "max_abs": float(difference.max()),
                    "rmse": float(difference.square().mean().sqrt()),
                    "exact": bool(torch.equal(first, second)),
                }
            )
        processed += len(ids)
        del features, chunks
        if sample_index % 16 == 15:
            snapshots.append(gpu_snapshot(f"sample_{sample_index + 1}"))
    elapsed = time.perf_counter() - started
    snapshots.append(gpu_snapshot("complete"))
    collector.close()
    payload = {
        "stage": "TD32-S1-EXTRACTOR-PROBE",
        "target": str(Path(args.target).resolve()),
        "source": str(Path(args.source).resolve()),
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(args.data),
        "token_budget": args.token_budget,
        "processed_tokens": processed,
        "union_taps": UNION_TAPS,
        "candidate_taps": TAP_CANDIDATES,
        "elapsed_seconds": elapsed,
        "tokens_per_second": processed / elapsed,
        "determinism": deterministic,
        "all_deterministic_exact": all(row["exact"] for row in deterministic),
        "per_tap": {},
        "gpu_preflight": preflight,
        "memory_snapshots": snapshots,
    }
    for tap, stat in sums.items():
        mean = stat["sum"] / stat["count"]
        variance = max(stat["sum2"] / stat["count"] - mean * mean, 0.0)
        payload["per_tap"][str(tap)] = {
            "elements": stat["count"],
            "mean": mean,
            "std": math.sqrt(variance),
            "min": stat["min"],
            "max": stat["max"],
        }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def flush_reference_shard(output_dir, shard_index, rows, parts):
    features = torch.cat(parts, dim=1).contiguous()
    payload = {
        "schema_version": 1,
        "taps": REFERENCE_TAPS,
        "rows": rows,
        "features": features,
    }
    path = output_dir / f"reference-{shard_index:04d}.pt"
    atomic_torch_save(path, payload)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "samples": len(rows),
        "tokens": int(features.shape[1]),
        "shape": list(features.shape),
    }


def run_reference(args):
    from e2e.speculative import SelectedHiddenCollector

    preflight = preflight_gpu()
    model = load_reference(args.reference)
    collector = SelectedHiddenCollector(model, REFERENCE_TAPS, folded_basis=True)
    snapshots = [gpu_snapshot("runtime_loaded")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards, rows, parts, shard_tokens = [], [], [], 0
    started, processed = time.perf_counter(), 0
    for sample_index, (row, ids) in enumerate(selected_samples(args.data, args.token_budget)):
        collector.reset()
        model_forward(model, ids)
        features = collector.features().detach().cpu().to(torch.bfloat16)
        if features.shape != (1, len(ids), 20480):
            raise RuntimeError(f"unexpected reference feature shape {features.shape}")
        if not torch.isfinite(features).all():
            raise RuntimeError("reference feature contains NaN or Inf")
        rows.append(
            {
                "sample_id": row["sample_id"],
                "token_ids_sha256": row["token_ids_sha256"],
                "tokens": len(ids),
            }
        )
        parts.append(features)
        shard_tokens += len(ids)
        processed += len(ids)
        if shard_tokens >= args.shard_tokens:
            shards.append(
                flush_reference_shard(output_dir, len(shards), rows, parts)
            )
            rows, parts, shard_tokens = [], [], 0
        if sample_index % 16 == 15:
            snapshots.append(gpu_snapshot(f"sample_{sample_index + 1}"))
    if parts:
        shards.append(flush_reference_shard(output_dir, len(shards), rows, parts))
    elapsed = time.perf_counter() - started
    snapshots.append(gpu_snapshot("complete"))
    collector.close()
    manifest = {
        "stage": "TD32-S2-REFERENCE",
        "reference": str(Path(args.reference).resolve()),
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(args.data),
        "taps": REFERENCE_TAPS,
        "token_budget": args.token_budget,
        "processed_tokens": processed,
        "elapsed_seconds": elapsed,
        "tokens_per_second": processed / elapsed,
        "shard_token_limit": args.shard_tokens,
        "shards": shards,
        "gpu_preflight": preflight,
        "memory_snapshots": snapshots,
    }
    atomic_json(output_dir / "reference_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def run_fit(args):
    from e2e.speculative import SelectedHiddenCollector, load_td_target_basis

    preflight = preflight_gpu()
    reference_manifest = json.loads(Path(args.reference_manifest).read_text())
    if reference_manifest["data_sha256"] != sha256_file(args.data):
        raise RuntimeError("reference/data SHA mismatch")
    target = load_quantized_target(args.target)
    signs, final_norm = load_td_target_basis(target, args.source)
    collector = SelectedHiddenCollector(
        target, UNION_TAPS, signs, final_norm, cache_basis=True
    )
    candidate_positions = {
        name: [UNION_TAPS.index(tap) for tap in taps]
        for name, taps in TAP_CANDIDATES.items()
    }
    stats = {name: SufficientStats(20480) for name in TAP_CANDIDATES}
    snapshots = [gpu_snapshot("runtime_loaded")]
    samples = iter(selected_samples(args.data, reference_manifest["token_budget"]))
    started, processed = time.perf_counter(), 0
    for shard_index, shard in enumerate(reference_manifest["shards"]):
        path = Path(shard["path"])
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"reference shard SHA mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        reference = payload["features"]
        offset = 0
        for expected in payload["rows"]:
            row, ids = next(samples)
            if row["sample_id"] != expected["sample_id"] or len(ids) != expected["tokens"]:
                raise RuntimeError("reference/input sample order mismatch")
            collector.reset()
            model_forward(target, ids)
            union = collector.features().detach()
            chunks = union.split(5120, dim=-1)
            y = reference[:, offset:offset + len(ids)].to("cuda").reshape(len(ids), 20480)
            for name, positions in candidate_positions.items():
                x = torch.cat([chunks[position] for position in positions], dim=-1)
                stats[name].update(x.reshape(len(ids), 20480), y)
            offset += len(ids)
            processed += len(ids)
            del union, chunks, y
        if offset != reference.shape[1]:
            raise RuntimeError("reference shard token count mismatch")
        del payload, reference
        gc.collect()
        torch.cuda.empty_cache()
        snapshots.append(gpu_snapshot(f"shard_{shard_index + 1}"))
    try:
        next(samples)
        raise RuntimeError("reference manifest did not consume all selected data")
    except StopIteration:
        pass
    elapsed = time.perf_counter() - started
    snapshots.append(gpu_snapshot("complete"))
    collector.close()
    output_dir = Path(args.output_dir)
    candidates = {}
    for name, candidate_stats in stats.items():
        scale, bias, metrics = candidate_stats.fit()
        calibration_path = output_dir / name / "calibration.pt"
        atomic_torch_save(
            calibration_path,
            {
                "raw_scale": scale,
                "raw_bias": bias,
                "source_split": "calibration_tune",
                "runtime": "fused_v1",
                "target_layers": TAP_CANDIDATES[name],
                "reference_layers": REFERENCE_TAPS,
                "target_model": "Qwen/Qwen3-32B-GPTQ-W4A4KV4",
                "reference_model": "Qwen/Qwen3-14B",
            },
        )
        candidate_manifest = {
            "name": name,
            "target_layers": TAP_CANDIDATES[name],
            "reference_layers": REFERENCE_TAPS,
            "calibration": str(calibration_path),
            "calibration_sha256": sha256_file(calibration_path),
            "metrics": metrics,
        }
        atomic_json(output_dir / name / "fit_metrics.json", candidate_manifest)
        candidates[name] = candidate_manifest
    manifest = {
        "stage": "TD32-S2-AFFINE-FIT",
        "target": str(Path(args.target).resolve()),
        "source": str(Path(args.source).resolve()),
        "data": str(Path(args.data).resolve()),
        "data_sha256": sha256_file(args.data),
        "reference_manifest": str(Path(args.reference_manifest).resolve()),
        "union_taps": UNION_TAPS,
        "processed_tokens": processed,
        "elapsed_seconds": elapsed,
        "tokens_per_second": processed / elapsed,
        "candidates": candidates,
        "gpu_preflight": preflight,
        "memory_snapshots": snapshots,
    }
    atomic_json(output_dir / "fit_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    s1 = subparsers.add_parser("probe")
    s1.add_argument("--target", required=True)
    s1.add_argument("--source", required=True)
    s1.add_argument("--data", required=True, type=Path)
    s1.add_argument("--token-budget", type=int, default=16000)
    s1.add_argument("--determinism-samples", type=int, default=4)
    s1.add_argument("--output", required=True, type=Path)
    s1.set_defaults(handler=run_s1)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--reference", required=True)
    reference.add_argument("--data", required=True, type=Path)
    reference.add_argument("--token-budget", type=int, default=128000)
    reference.add_argument("--shard-tokens", type=int, default=16000)
    reference.add_argument("--output-dir", required=True)
    reference.set_defaults(handler=run_reference)

    fit = subparsers.add_parser("fit")
    fit.add_argument("--target", required=True)
    fit.add_argument("--source", required=True)
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--reference-manifest", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.set_defaults(handler=run_fit)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
