"""Run and compare leakage-free heldout AR/TD32 S2 calibration smokes."""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch

from e2e.qwen3_32b_td_features import TAP_CANDIDATES


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


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
        raise RuntimeError(f"GPU is not idle: {used / 2**20:.1f} MiB")
    return {"device_total_bytes": int(total), "device_used_bytes": int(used)}


def override_td_taps(runtime, taps):
    from e2e.speculative import SelectedHiddenCollector

    old = runtime.collector
    if old is None:
        raise RuntimeError("TD runtime has no selected-hidden collector")
    signs, final_norm = old.rotation_signs, old.final_norm_weight
    old.close()
    runtime.spec = replace(runtime.spec, target_layers=tuple(taps))
    runtime.collector = SelectedHiddenCollector(
        runtime.target,
        taps,
        signs,
        final_norm,
        cache_basis=True,
    )


def run(args):
    from e2e.speculative import load_runtime

    if args.mode == "td" and not args.candidate:
        raise ValueError("TD mode requires --candidate")
    if args.mode in ("ar", "ti") and (args.candidate or args.calibration):
        raise ValueError(f"{args.mode.upper()} mode must not receive TD candidate/calibration")
    rows = read_jsonl(args.data)
    if not rows:
        raise RuntimeError("heldout data is empty")
    preflight = preflight_gpu()
    mode = {"ar": "ar", "td": "pard2-td", "ti": "pard2-ti"}[args.mode]
    runtime = load_runtime(
        mode=mode,
        target_checkpoint=args.target,
        draft_snapshot=None if mode == "ar" else args.draft,
        tokenizer_path=args.tokenizer,
        max_cache_len=args.max_cache_len,
        compile_mode="eager",
        calibration_path=args.calibration,
        td_proxy_profile=("qwen3-14b-on-qwen3-32b" if mode == "pard2-td" else None),
        exact_row_norm=True,
        rowwise_lm_head=False,
        td_cache_basis=True,
        td_lazy_features=False,
        td_unique_projection=False,
        td_basis_fold=False,
        fused_norm_quant=False,
    )
    if mode == "pard2-td":
        override_td_taps(runtime, TAP_CANDIDATES[args.candidate])
    snapshots = [gpu_snapshot("runtime_loaded")]
    warmup_ids = torch.tensor(
        [rows[0]["input_ids"]], device="cuda", dtype=torch.long
    )
    runtime.generate(warmup_ids, min(args.generated_tokens, 8))
    gc.collect()
    torch.cuda.empty_cache()
    snapshots.append(gpu_snapshot("warmup_complete"))
    runs = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        ids = torch.tensor([row["input_ids"]], device="cuda", dtype=torch.long)
        result = runtime.generate(ids, args.generated_tokens)
        memory = gpu_snapshot(f"sample_{index + 1}")
        metrics = result.metrics()
        runs.append(
            {
                "sample_id": row["sample_id"],
                "stratum": row["stratum"],
                "input_token_count": int(ids.numel()),
                "input_token_sha256": row["token_ids_sha256"],
                "output_ids": result.output_ids,
                **metrics,
                "memory_snapshot": memory,
            }
        )
        gc.collect()
        torch.cuda.empty_cache()
    elapsed = time.perf_counter() - started
    snapshots.append(gpu_snapshot("complete"))
    runtime.close()
    payload = {
        "stage": "TD32-S2-HELDOUT",
        "mode": args.mode,
        "candidate": args.candidate,
        "target_layers": (
            TAP_CANDIDATES[args.candidate] if args.candidate else None
        ),
        "target": str(Path(args.target).resolve()),
        "draft": str(Path(args.draft).resolve()) if args.draft else None,
        "calibration": (
            str(Path(args.calibration).resolve()) if args.calibration else None
        ),
        "calibration_sha256": (
            sha256_file(Path(args.calibration)) if args.calibration else None
        ),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "generated_tokens": args.generated_tokens,
        "max_cache_len": args.max_cache_len,
        "elapsed_seconds": elapsed,
        "median_steady_tokens_per_s": statistics.median(
            row["steady_tokens_per_s"] for row in runs
        ),
        "median_end_to_end_tokens_per_s": statistics.median(
            row["end_to_end_tokens_per_s"] for row in runs
        ),
        "runs": runs,
        "gpu_preflight": preflight,
        "memory_snapshots": snapshots,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "runs"},
            indent=2,
        )
    )


def compare(args):
    ar = json.loads(args.ar.read_text())
    if ar["mode"] != "ar":
        raise ValueError("--ar input is not AR")
    ar_rows = {row["sample_id"]: row for row in ar["runs"]}
    candidates = {}
    winner = None
    for path in args.td:
        td = json.loads(path.read_text())
        name = td["candidate"]
        if not name or name in candidates:
            raise ValueError(f"invalid/duplicate TD candidate {name!r}")
        paired = []
        per_stratum = {}
        parity = True
        for row in td["runs"]:
            reference = ar_rows.get(row["sample_id"])
            if reference is None:
                raise RuntimeError("TD heldout row is absent from AR result")
            same = row["output_ids"] == reference["output_ids"]
            parity = parity and same
            steady_ratio = row["steady_tokens_per_s"] / reference["steady_tokens_per_s"]
            e2e_ratio = row["end_to_end_tokens_per_s"] / reference["end_to_end_tokens_per_s"]
            paired.append((steady_ratio, e2e_ratio))
            group = per_stratum.setdefault(
                row["stratum"],
                {"runs": 0, "accepted": 0, "proposed": 0, "accept_lengths": []},
            )
            group["runs"] += 1
            group["accepted"] += row["accepted_draft_tokens"]
            group["proposed"] += row["proposed_draft_tokens"]
            group["accept_lengths"].extend(row["accept_length_by_step"])
        all_accept_lengths = [
            value
            for group in per_stratum.values()
            for value in group["accept_lengths"]
        ]
        mean_accept = statistics.mean(all_accept_lengths)
        for group in per_stratum.values():
            group["mean_accept_length"] = statistics.mean(group["accept_lengths"])
            del group["accept_lengths"]
        gate = {
            "exact_ar_parity": parity,
            "all_strata_nonzero_accepted": all(
                group["accepted"] > 0 for group in per_stratum.values()
            ),
            "aggregate_mean_accept_at_least_2": mean_accept >= 2.0,
        }
        gate["pass"] = all(gate.values())
        candidates[name] = {
            "path": str(path.resolve()),
            "target_layers": td["target_layers"],
            "mean_accept_length": mean_accept,
            "paired_median_steady_ratio": statistics.median(x[0] for x in paired),
            "paired_median_end_to_end_ratio": statistics.median(x[1] for x in paired),
            "per_stratum": per_stratum,
            "gate": gate,
        }
    passing = [
        (name, value)
        for name, value in candidates.items()
        if value["gate"]["pass"]
    ]
    if passing:
        winner = max(
            passing,
            key=lambda item: (
                item[1]["paired_median_end_to_end_ratio"],
                item[1]["mean_accept_length"],
            ),
        )[0]
    payload = {
        "stage": "TD32-S2-DECISION",
        "ar": str(args.ar.resolve()),
        "candidates": candidates,
        "winner": winner,
        "stage_pass": winner is not None,
        "selection_metric": "passing gate, then paired median E2E, then mean acceptance",
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mode", choices=("ar", "td", "ti"), required=True)
    run_parser.add_argument("--candidate", choices=tuple(TAP_CANDIDATES))
    run_parser.add_argument("--target", required=True)
    run_parser.add_argument("--draft")
    run_parser.add_argument("--tokenizer", required=True)
    run_parser.add_argument("--calibration")
    run_parser.add_argument("--data", type=Path, required=True)
    run_parser.add_argument("--generated-tokens", type=int, default=32)
    run_parser.add_argument("--max-cache-len", type=int, default=2048)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(handler=run)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--ar", type=Path, required=True)
    compare_parser.add_argument("--td", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.set_defaults(handler=compare)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
