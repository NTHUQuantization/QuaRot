"""Actual target launches, fixed-context latency and strict cache/causal checks.

Run through run_ab.py so every optimization switch is explicitly recorded.
No profiler is active during event or wall-clock latency sampling.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch
import quarot
from e2e.benchmark_pard2 import preflight_gpu
from e2e.pard2 import DEFAULT_TARGET, DEFAULT_TOKENIZER, verify_target_checkpoint
from e2e.speculative import load_runtime


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slots(cache, start, count):
    """Only meaningful initialized slots; never compare stale unused storage."""
    result = {}
    for name in ("pages", "scales"):
        storage = getattr(cache, name)
        result[name] = torch.stack([
            storage[position // cache.page_size, ..., position % cache.page_size, :]
            for position in range(start, start + count)
        ]).detach().cpu()
    return result


def exact_cache(left, right):
    return {name: torch.equal(left[name], right[name]) for name in left}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_TARGET))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--contexts", nargs="+", type=int, default=[127, 128, 129])
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 15, 16])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--fused-norm-quant", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = preflight_gpu()
    verify_target_checkpoint(args.model, "qwen3_8b")
    torch.manual_seed(20260924)
    torch.set_num_threads(4)
    runtime = load_runtime(
        mode="ar", target_checkpoint=args.model, draft_snapshot=None,
        tokenizer_path=args.tokenizer, max_cache_len=max(args.contexts) + 64,
        compile_mode="eager", fused_norm_quant=args.fused_norm_quant,
        benchmark_profile="qwen3_8b")
    result = {
        "scope": "target-only verification shape; exact-row RMSNorm; no drafter/TD collector",
        "gpu_preflight": gpu, "model": args.model,
        "torch": torch.__version__, "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(0),
        "extension_sha256": digest(quarot._HIP.__file__),
        "flags": {k: v for k, v in os.environ.items() if k.startswith("QUAROT_")},
        "fused_norm_quant": args.fused_norm_quant,
        "warmups": args.warmups, "samples": args.samples, "cases": {},
        "standards": {
            "cache": "bit-exact packed KV and FP16 scale/zero",
            "causal": "unchanged earlier-row logits when future input changes",
            "tokens": "exact argmax and end-to-end generated token parity",
            "logits": "record max error; no new relaxed tolerance is introduced",
        },
    }
    artifacts = {}
    reference = (torch.load(args.reference, map_location="cpu", weights_only=True)
                 if args.reference else None)
    all_checks = []

    def write():
        (args.output / "target.json").write_text(json.dumps(result, indent=2) + "\n")

    with torch.inference_mode():
        for context in args.contexts:
            prefix = torch.randint(100, 10000, (1, context), device="cuda")
            candidates = torch.randint(100, 10000, (1, max(args.rows)), device="cuda")
            cache = runtime._target_cache()
            out, _ = runtime._target_call(prefix, cache, torch.arange(context, device="cuda"))
            cache = out.past_key_values
            del out

            def call(ids, base=context):
                cache.length = base
                positions = torch.arange(base, base + ids.shape[1], device="cuda")
                return runtime._target_call(ids, cache, positions)[0]

            positions_by_rows = {
                rows: torch.arange(context, context + rows, device="cuda")
                for rows in args.rows}
            for rows in args.rows:
                key = f"context{context}_M{rows}"
                ids = candidates[:, :rows]
                for _ in range(args.warmups):
                    call(ids)
                torch.cuda.synchronize()
                cache.length = context
                with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA]) as prof:
                    out, _ = runtime._target_call(ids, cache, positions_by_rows[rows])
                    torch.cuda.synchronize()
                trace = args.output / f"{key}.trace.json"
                prof.export_chrome_trace(str(trace))
                events = json.loads(trace.read_text())["traceEvents"]
                kernels = [event for event in events if event.get("cat") == "kernel"]
                replayed = getattr(runtime, "_verification_graphs", {}).get((id(cache), rows))
                if not kernels and replayed is None:
                    raise RuntimeError("profiler emitted no GPU kernel events; launch count is unavailable")
                launch_apis = [event for event in events
                               if event.get("cat") in ("cuda_runtime", "hip_runtime")
                               and "launch" in event.get("name", "").lower()]
                case = {
                    "context": context, "rows": rows,
                    "all_kernel_launches": len(kernels),
                    "trace_gpu_kernel_events": len(kernels),
                    "kernel_count_source": "GPU trace",
                    "cpu_launch_api_calls": len(launch_apis),
                    "cpu_launch_api_histogram": dict(Counter(event["name"] for event in launch_apis)),
                    "gpu_memcpy_events": sum(event.get("cat") == "gpu_memcpy" for event in events),
                    "kernel_duration_sum_us": sum(event["dur"] for event in kernels),
                    "kernel_histogram": dict(Counter(event["name"] for event in kernels)),
                    "event_ms": [], "wall_ms": [],
                    "verification_graph_captures": [
                        {"rows": graph_key[1], "capture_ms": graph.capture_ms,
                         "node_metadata": graph.node_metadata}
                        for graph_key, graph in getattr(runtime, "_verification_graphs", {}).items()],
                }
                if replayed is not None:
                    # ROCTracer 7.2 emits only a variable subset of graph
                    # child events. Count instantiated kernel nodes instead;
                    # external scalar fill is an ordinary launch API event.
                    direct_names = {"hipLaunchKernel", "hipExtModuleLaunchKernel",
                                    "hipModuleLaunchKernel", "hipExtLaunchKernel"}
                    direct = sum(event["name"] in direct_names for event in launch_apis)
                    case["actual_graph_kernel_nodes"] = replayed.node_metadata["kernel_nodes"]
                    case["external_kernel_launches"] = direct
                    case["all_kernel_launches"] = replayed.node_metadata["kernel_nodes"] + direct
                    case["kernel_count_source"] = "HIP graph kernel nodes + external kernel launch APIs"
                    case["trace_gpu_kernel_events_complete"] = False
                    case["trace_kernel_duration_sum_us"] = case["kernel_duration_sum_us"]
                    case["kernel_duration_sum_us"] = None
                    case["kernel_histogram_source"] = "incomplete graph replay GPU trace"
                result["cases"][key] = case
                # Reset and input-position allocation are excluded just as in
                # the original audit and runtime target_verify stage timer.
                for _ in range(args.samples):
                    cache.length = context
                    torch.cuda.synchronize()
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    before = time.perf_counter()
                    start.record()
                    out, _ = runtime._target_call(ids, cache, positions_by_rows[rows])
                    end.record()
                    end.synchronize()
                    case["wall_ms"].append((time.perf_counter() - before) * 1000)
                    case["event_ms"].append(start.elapsed_time(end))
                for name in ("event_ms", "wall_ms"):
                    case[name + "_median"] = statistics.median(case[name])
                # Graph outputs are persistent storage: clone before replay.
                artifact = {"logits": out.logits.detach().cpu().clone(),
                            **slots(cache, context, rows)}
                artifacts[key] = artifact
                checks = {"finite_logits": bool(torch.isfinite(artifact["logits"]).all())}
                if reference is not None:
                    expected = reference[key]
                    checks.update({"baseline_" + name: equal for name, equal in
                                   exact_cache({k: artifact[k] for k in ("pages", "scales")},
                                               expected).items()})
                    checks["baseline_argmax"] = torch.equal(
                        artifact["logits"].argmax(-1), expected["logits"].argmax(-1))
                    case["baseline_logits_max_abs"] = float(
                        (artifact["logits"].float() - expected["logits"].float()).abs().max())
                    case["baseline_logits_bit_exact"] = torch.equal(
                        artifact["logits"], expected["logits"])
                if not args.skip_correctness:
                    sequential = []
                    for row in range(rows):
                        single = call(ids[:, row:row + 1], context + row)
                        sequential.append(single.logits.detach().cpu().clone())
                    seq_logits = torch.cat(sequential, dim=1)
                    checks.update({"sequential_" + name: equal for name, equal in
                                   exact_cache(artifact, {"logits": seq_logits,
                                       **slots(cache, context, rows)}).items()
                                   if name != "logits"})
                    checks["sequential_argmax"] = torch.equal(
                        artifact["logits"].argmax(-1), seq_logits.argmax(-1))
                    case["sequential_logits_max_abs"] = float(
                        (artifact["logits"].float() - seq_logits.float()).abs().max())
                    if rows > 1:
                        altered = ids.clone()
                        altered[:, -1] += 1
                        changed = call(altered).logits.detach().cpu().clone()
                        checks["causal_future_token"] = torch.equal(
                            artifact["logits"][:, :-1], changed[:, :-1])
                    # Explicit reject-all and partial-accept rollback; stale
                    # slots must not affect replacement at a crossed page.
                    for keep in sorted({0, rows // 2}):
                        call(ids)
                        correction = (ids[:, :1] + 19).contiguous()
                        after_reject = call(correction, context + keep)
                        rejected_logits = after_reject.logits.detach().cpu().clone()
                        rejected_slot = slots(cache, context + keep, 1)
                        if keep:
                            call(ids[:, :keep])
                        clean = call(correction, context + keep)
                        checks[f"rollback{keep}_logits"] = torch.equal(
                            rejected_logits, clean.logits.detach().cpu())
                        checks.update({f"rollback{keep}_{name}": equal for name, equal in
                                       exact_cache(rejected_slot,
                                                   slots(cache, context + keep, 1)).items()})
                case["checks"] = checks
                all_checks.extend(checks.values())
                print(key, case["all_kernel_launches"],
                      case["event_ms_median"], checks, flush=True)
                write()
            # Graph runtimes may deliberately reuse a cache and its buffers.
            del cache
        result["passed"] = all(all_checks)
        torch.save(artifacts, args.output / "artifacts.pt")
        write()
    runtime.close()
    if not result["passed"]:
        raise SystemExit("strict correctness check failed; see target.json")


if __name__ == "__main__":
    main()
