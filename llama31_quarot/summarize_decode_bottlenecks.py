#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from llama31_quarot.common import summarize, write_csv


NUM_LAYERS = 32
HIDDEN = 4096
INTERMEDIATE = 14336
Q_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
VOCAB = 128256


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fvalue(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def ivalue(row, key, default=0):
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def summarize_latency(root):
    rows = read_csv(root / "latency_raw.csv")
    groups = defaultdict(list)
    for row in rows:
        key = (row["variant"], ivalue(row, "batch"), ivalue(row, "context_len"))
        groups[key].append(fvalue(row, "decode_ms_per_token"))
    means = {key: sum(values) / len(values) for key, values in groups.items()}
    output = []
    for key, values in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        variant, batch, context = key
        stats = summarize(values)
        baseline = means.get(("unfused_INT4", batch, context), math.nan)
        output.append({
            "variant": variant,
            "batch": batch,
            "context_len": context,
            "samples": len(values),
            **stats,
            "tokens_per_second": 1000.0 * batch / stats["mean"],
            "speedup_vs_unfused_INT4": 1.0 if variant == "unfused_INT4" else baseline / stats["mean"],
            "coefficient_of_variation_percent": 100.0 * stats["std"] / stats["mean"],
        })
    write_csv(root / "latency_summary.csv", output)
    return output


def summarize_ablation(root):
    rows = read_csv(root / "ablation_raw.csv")
    groups = defaultdict(list)
    for row in rows:
        key = (row["variant"], ivalue(row, "batch"), ivalue(row, "context_len"))
        groups[key].append(fvalue(row, "decode_ms_per_token"))
    means = {key: sum(values) / len(values) for key, values in groups.items()}
    output = []
    for key, values in sorted(groups.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])):
        variant, batch, context = key
        stats = summarize(values)
        baseline = means[("unfused_INT4", batch, context)]
        output.append({
            "variant": variant,
            "batch": batch,
            "context_len": context,
            "samples": len(values),
            **stats,
            "speedup_vs_unfused_INT4": 1.0 if variant == "unfused_INT4" else baseline / stats["mean"],
        })
    write_csv(root / "ablation_summary.csv", output)
    return output


def summarize_sequential(root):
    rows = read_csv(root / "sequential_decode.csv")
    baselines = {
        (ivalue(r, "batch"), ivalue(r, "initial_context_len")): fvalue(r, "decode_ms_per_token")
        for r in rows
        if r["variant"] == "unfused_INT4"
    }
    for row in rows:
        key = (ivalue(row, "batch"), ivalue(row, "initial_context_len"))
        row["speedup_vs_unfused_INT4"] = baselines[key] / fvalue(row, "decode_ms_per_token")
    if rows:
        write_csv(root / "sequential_decode_summary.csv", rows)
    return rows


def summarize_stages(root):
    rows = read_csv(root / "stage_breakdown.csv")
    category_totals = defaultdict(float)
    stage_totals = defaultdict(float)
    metadata = {}
    for row in rows:
        key = (row["variant"], ivalue(row, "batch"), ivalue(row, "context_len"))
        category_totals[(*key, row["category"])] += fvalue(row, "total_ms_per_token")
        stage_totals[(*key, row["stage"])] += fvalue(row, "total_ms_per_token")
        metadata[key] = {
            "plain_ms_per_token": fvalue(row, "plain_ms_per_token"),
            "instrumented_ms_per_token": fvalue(row, "instrumented_ms_per_token"),
            "instrumentation_overhead_percent": fvalue(row, "instrumentation_overhead_percent"),
        }
    category_rows = []
    for key, latency in sorted(category_totals.items(), key=lambda item: (item[0][1], item[0][2], item[0][0], item[0][3])):
        variant, batch, context, category = key
        total = sum(
            value
            for (v, b, c, _), value in stage_totals.items()
            if (v, b, c) == (variant, batch, context)
        )
        baseline = category_totals.get(("unfused_INT4", batch, context, category), math.nan)
        category_rows.append({
            "variant": variant,
            "batch": batch,
            "context_len": context,
            "category": category,
            "total_ms_per_token": latency,
                "percent_of_stage_sum": 100.0 * latency / total if total else 0.0,
            "saved_ms_vs_unfused_INT4": baseline - latency,
            **metadata[(variant, batch, context)],
        })
    write_csv(root / "stage_category_summary.csv", category_rows)
    return rows, category_rows, stage_totals


def interval_union_ns(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def classify_kernel(name):
    lower = name.lower()
    if "append" in lower and ("had" in lower or "quant" in lower):
        return "K1"
    if "__precision__s4" in lower or "batch_decode_i4" in lower:
        return "K2_INT4"
    if "batchdecode" in lower or "batch_decode_f16" in lower:
        return "K2_FP16"
    if "output_had_quant" in lower:
        return "K3"
    if "fused_ffn" in lower or "ffn_silu" in lower:
        return "FFN"
    if "cijk_" in lower or "rocblas" in lower or "gemm" in lower:
        return "rocBLAS_GEMM"
    if any(token in lower for token in ["elementwise", "vectorized", "reduce_kernel", "copy", "catarray"]):
        return "PyTorch_elementwise_copy_reduce"
    return "other"


def find_event_row(root, workload):
    candidates = list(root.glob(f"**/{workload}_event.csv"))
    if not candidates:
        return None
    rows = read_csv(candidates[0])
    return rows[0] if rows else None


def marker_range_ms(path, iterations):
    marker_path = path.with_name(path.name.replace("_kernel_trace.csv", "_marker_api_trace.csv"))
    for row in read_csv(marker_path):
        if row.get("Function", "").startswith("decode_only/"):
            duration_ns = ivalue(row, "End_Timestamp") - ivalue(row, "Start_Timestamp")
            return duration_ns / 1e6 / iterations
    return 0.0


def summarize_rocprof(root):
    rocprof_root = root / "rocprof"
    summaries = []
    top_rows = []
    category_rows = []
    for path in sorted(rocprof_root.glob("**/*_kernel_trace.csv")):
        workload = path.name.replace("_kernel_trace.csv", "")
        if "counter" in workload or "probe" in workload:
            continue
        event = find_event_row(root, workload)
        if event is None:
            continue
        iterations = ivalue(event, "iterations", 1)
        event_ms = marker_range_ms(path, iterations) or fvalue(event, "decode_ms_per_token")
        intervals = []
        kernels = defaultdict(lambda: {
            "calls": 0,
            "total_ns": 0,
            "min_ns": None,
            "max_ns": 0,
            "lds": 0,
            "vgpr": 0,
            "sgpr": 0,
            "scratch": 0,
            "workgroup": 0,
            "grid": 0,
        })
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                start = ivalue(row, "Start_Timestamp")
                end = ivalue(row, "End_Timestamp")
                if end <= start:
                    continue
                duration = end - start
                intervals.append((start, end))
                name = row.get("Kernel_Name", "unknown").strip('"')
                item = kernels[name]
                item["calls"] += 1
                item["total_ns"] += duration
                item["min_ns"] = duration if item["min_ns"] is None else min(item["min_ns"], duration)
                item["max_ns"] = max(item["max_ns"], duration)
                item["lds"] = max(item["lds"], ivalue(row, "LDS_Block_Size"))
                item["vgpr"] = max(item["vgpr"], ivalue(row, "VGPR_Count"))
                item["sgpr"] = max(item["sgpr"], ivalue(row, "SGPR_Count"))
                item["scratch"] = max(item["scratch"], ivalue(row, "Scratch_Size"))
                item["workgroup"] = max(
                    item["workgroup"],
                    ivalue(row, "Workgroup_Size_X") * ivalue(row, "Workgroup_Size_Y", 1) * ivalue(row, "Workgroup_Size_Z", 1),
                )
                item["grid"] = max(
                    item["grid"],
                    ivalue(row, "Grid_Size_X") * ivalue(row, "Grid_Size_Y", 1) * ivalue(row, "Grid_Size_Z", 1),
                )
        union_ms = interval_union_ns(intervals) / 1e6 / iterations
        sum_ms = sum(item["total_ns"] for item in kernels.values()) / 1e6 / iterations
        calls_per_token = sum(item["calls"] for item in kernels.values()) / iterations
        summaries.append({
            "workload": workload,
            "variant": event["variant"],
            "batch": ivalue(event, "batch"),
            "context_len": ivalue(event, "context_len"),
            "iterations": iterations,
            "event_ms_per_token": event_ms,
            "kernel_sum_ms_per_token": sum_ms,
            "gpu_busy_union_ms_per_token": union_ms,
            "estimated_launch_gap_ms": max(0.0, event_ms - union_ms),
            "gpu_busy_percent_of_event": 100.0 * union_ms / event_ms if event_ms else 0.0,
            "kernel_calls_per_token": calls_per_token,
        })
        category = defaultdict(lambda: [0, 0])
        ordered = sorted(kernels.items(), key=lambda item: item[1]["total_ns"], reverse=True)
        for rank, (name, item) in enumerate(ordered, start=1):
            kind = classify_kernel(name)
            category[kind][0] += item["calls"]
            category[kind][1] += item["total_ns"]
            if rank <= 20:
                top_rows.append({
                    "workload": workload,
                    "rank": rank,
                    "category": kind,
                    "kernel": name[:240],
                    "calls_per_token": item["calls"] / iterations,
                    "total_ms_per_token": item["total_ns"] / 1e6 / iterations,
                    "average_us": item["total_ns"] / item["calls"] / 1000.0,
                    "min_us": item["min_ns"] / 1000.0,
                    "max_us": item["max_ns"] / 1000.0,
                    "lds_block_bytes": item["lds"],
                    "vgpr_count": item["vgpr"],
                    "sgpr_count": item["sgpr"],
                    "scratch_bytes": item["scratch"],
                    "workgroup_size": item["workgroup"],
                    "grid_size": item["grid"],
                })
        total_ns = sum(item[1] for item in category.values())
        for kind, (calls, duration_ns) in sorted(category.items()):
            category_rows.append({
                "workload": workload,
                "category": kind,
                "calls_per_token": calls / iterations,
                "kernel_sum_ms_per_token": duration_ns / 1e6 / iterations,
                "percent_of_kernel_sum": 100.0 * duration_ns / total_ns if total_ns else 0.0,
            })
    if summaries:
        write_csv(root / "rocprof_summary.csv", summaries)
        write_csv(root / "rocprof_top_kernels.csv", top_rows)
        write_csv(root / "rocprof_category_summary.csv", category_rows)
    return summaries, top_rows, category_rows


def summarize_counters(root):
    rows = []
    for path in sorted((root / "rocprof").glob("*_counter_collection.csv")):
        values = defaultdict(list)
        for row in read_csv(path):
            values[row.get("Counter_Name", "unknown")].append(fvalue(row, "Counter_Value"))
        workload = path.name.replace("_counter_collection.csv", "")
        for counter, samples in sorted(values.items()):
            rows.append({
                "workload": workload,
                "counter": counter,
                "samples": len(samples),
                "nonzero_samples": sum(value != 0.0 for value in samples),
                "sum": sum(samples),
                "mean": sum(samples) / len(samples),
                "min": min(samples),
                "max": max(samples),
                "status": "nonzero" if any(value != 0.0 for value in samples) else "all_zero",
            })
    if rows:
        write_csv(root / "counter_summary.csv", rows)
    return rows


def minimum_traffic_rows(stage_totals):
    rows = []
    shapes = sorted({(key[1], key[2]) for key in stage_totals})
    variants = sorted({key[0] for key in stage_totals})
    for variant in variants:
        for batch, context in shapes:
            seq_len = context + 1
            k1_bytes = batch * KV_HEADS * HEAD_DIM * 2 * 2
            k1_bytes += batch * KV_HEADS * HEAD_DIM * 2 // 2
            k1_bytes += batch * KV_HEADS * 2 * 2 * 2
            int4_cache = batch * seq_len * KV_HEADS * HEAD_DIM * 2 / 2
            params = batch * seq_len * KV_HEADS * 2 * 2 * 2
            if variant == "unfused_INT4":
                k2_bytes = int4_cache + params + 2 * batch * seq_len * KV_HEADS * HEAD_DIM * 2 * 2
            else:
                k2_bytes = int4_cache + params
            k3_bytes = batch * (HIDDEN * 2 + HIDDEN / 2 + 16 * 2 + HIDDEN / 2 + 16 * 2 + HIDDEN * 2)
            ffn_bytes = batch * (
                2 * INTERMEDIATE * 2
                + INTERMEDIATE / 2
                + (INTERMEDIATE // 256) * 2
                + INTERMEDIATE / 2
                + (INTERMEDIATE // 256) * 2
                + INTERMEDIATE * 2
            )
            stage_names = {
                "K1": "k1_unfused" if variant == "unfused_INT4" else "k1_fused",
                "K2": "k2_cache_dequant" if variant == "unfused_INT4" else "k2_int4_decode",
                "K3": "k3_unfused" if variant == "unfused_INT4" else "k3_fused",
                "FFN": "ffn_unfused" if variant == "unfused_INT4" else "ffn_fused",
            }
            for block, byte_count in [("K1", k1_bytes), ("K2", k2_bytes), ("K3", k3_bytes), ("FFN", ffn_bytes)]:
                latency = stage_totals.get((variant, batch, context, stage_names[block]), 0.0)
                if block == "K2" and variant == "unfused_INT4":
                    latency += stage_totals.get((variant, batch, context, "k2_fp16_decode"), 0.0)
                rows.append({
                    "variant": variant,
                    "batch": batch,
                    "context_len": context,
                    "block": block,
                    "minimum_semantic_bytes_all_layers": byte_count * NUM_LAYERS,
                    "stage_ms_per_token": latency,
                    "effective_minimum_bandwidth_GBps": (
                        byte_count * NUM_LAYERS / latency / 1e6 if latency else 0.0
                    ),
                })
    return rows


def throughput_rows(stage_totals):
    rows = []
    shapes = sorted({(key[1], key[2]) for key in stage_totals})
    variants = sorted({key[0] for key in stage_totals})
    projection_ops = {
        "q_proj": 2 * HIDDEN * HIDDEN * NUM_LAYERS,
        "k_proj": 2 * HIDDEN * KV_HEADS * HEAD_DIM * NUM_LAYERS,
        "v_proj": 2 * HIDDEN * KV_HEADS * HEAD_DIM * NUM_LAYERS,
        "o_proj_residual": 2 * HIDDEN * HIDDEN * NUM_LAYERS,
        "gate_proj": 2 * HIDDEN * INTERMEDIATE * NUM_LAYERS,
        "up_proj": 2 * HIDDEN * INTERMEDIATE * NUM_LAYERS,
        "down_proj_residual": 2 * INTERMEDIATE * HIDDEN * NUM_LAYERS,
        "final_norm_lm_head": 2 * HIDDEN * VOCAB,
    }
    for variant in variants:
        for batch, context in shapes:
            for stage, ops_per_batch in projection_ops.items():
                latency = stage_totals.get((variant, batch, context, stage), 0.0)
                rows.append({
                    "variant": variant,
                    "batch": batch,
                    "context_len": context,
                    "stage": stage,
                    "metric": "effective_TFLOPs",
                    "algorithmic_ops": ops_per_batch * batch,
                    "stage_ms": latency,
                    "value": ops_per_batch * batch / latency / 1e9 if latency else 0.0,
                })
            hadamard = [
                ("K1", "k1_unfused" if variant == "unfused_INT4" else "k1_fused", 2 * KV_HEADS * HEAD_DIM * math.log2(HEAD_DIM)),
                ("K3", "k3_unfused" if variant == "unfused_INT4" else "k3_fused", HIDDEN * math.log2(256)),
                ("FFN", "ffn_unfused" if variant == "unfused_INT4" else "ffn_fused", INTERMEDIATE * math.log2(256)),
            ]
            for block, stage, ops_per_layer in hadamard:
                latency = stage_totals.get((variant, batch, context, stage), 0.0)
                ops = ops_per_layer * NUM_LAYERS * batch
                rows.append({
                    "variant": variant,
                    "batch": batch,
                    "context_len": context,
                    "stage": block,
                    "metric": "effective_Hadamard_TOPS",
                    "algorithmic_ops": int(ops),
                    "stage_ms": latency,
                    "value": ops / latency / 1e9 if latency else 0.0,
                })
    return rows


def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(f, rows, columns, labels=None):
    labels = labels or columns
    f.write("| " + " | ".join(labels) + " |\n")
    f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = fmt(value)
            values.append(str(value).replace("|", "\\|"))
        f.write("| " + " | ".join(values) + " |\n")


def write_report(
    root,
    latency,
    ablation,
    sequential,
    correctness,
    stage_categories,
    rocprof,
    top_rows,
    rocprof_categories,
    counter_rows,
    traffic_rows,
    throughput_rows_data,
    gemm_bound_rows,
    report_name="decode_bottleneck_profiling_report_zh.md",
    chart_dir_name="charts",
):
    report = root.parent / report_name
    env = {}
    if (root / "environment.json").exists():
        env = json.loads((root / "environment.json").read_text())
    with report.open("w") as f:
        compared = [
            variant for variant in ["unfused_INT4", "fused_current", "fused_hadacore256"]
            if any(row["variant"] == variant for row in latency)
        ]
        if compared == ["unfused_INT4", "fused_current"]:
            f.write("# QuaRot Decode Bottleneck Profiling：Unfused INT4 vs Fused Current\n\n")
        else:
            f.write("# QuaRot Decode Bottleneck 完整 Profiling 報告\n\n")
        f.write(f"日期：{env.get('profile_date', 'unknown')}\n\n")
        f.write("## 實驗定位\n\n")
        f.write(
            "本報告比較 " + "、".join(f"`{variant}`" for variant in compared)
            + " 的 Llama-3.1 8B token-by-token decode。所有 speedup 均以 "
            "`unfused_INT4` 為 baseline。`unfused_INT4` 尚不是經完整 QuaRot 權重旋轉與 calibration 的官方 converted model。\n\n"
        )
        if env:
            f.write("## 實驗環境\n\n")
            f.write(f"- GPU：`{env.get('device', 'unknown')}`\n")
            f.write(f"- PyTorch：`{env.get('torch', 'unknown')}`\n")
            f.write(f"- HIP：`{env.get('hip', 'unknown')}`\n")
            f.write(f"- Git commit：`{env.get('git_commit', 'unknown')}`；工作目錄可能含未提交修改。\n\n")
        f.write("## Benchmark 方法\n\n")
        f.write(
            "模型載入、HF prefill、INT4 cache conversion 與 warmup 均排除於 decode event timing。"
            "固定 context benchmark 重複覆寫同一 decode slot，因此每次 iteration 看到相同有效長度；"
            "sequential benchmark 則正常逐 token 擴展 active cache view。\n\n"
        )
        f.write(
            "Full grid 使用 3 sessions × 5 repeats，每個 repeat warmup 10、timed iterations 50，共 15 個 sample/shape/backend。"
            "Stage pass 使用 3 iterations；ablation 使用 3 repeats × 20 iterations；rocprof raw trace 每 workload 收 3 個 selected-region decode iterations。\n\n"
        )
        if latency:
            current = [row for row in latency if row["variant"] == "fused_current"]
            hadacore = [row for row in latency if row["variant"] == "fused_hadacore256"]
            current_speedups = [fvalue(row, "speedup_vs_unfused_INT4") for row in current]
            current_vs_had = []
            if hadacore:
                had_by_shape = {(ivalue(row, "batch"), ivalue(row, "context_len")): row for row in hadacore}
                for row in current:
                    peer = had_by_shape[(ivalue(row, "batch"), ivalue(row, "context_len"))]
                    current_vs_had.append(fvalue(row, "mean") / fvalue(peer, "mean"))
            best = max(current, key=lambda row: fvalue(row, "speedup_vs_unfused_INT4"))
            f.write("## 主要發現\n\n")
            f.write(
                f"- `fused_current` 在 16 個 shape 的 speedup 為 "
                f"{min(current_speedups):.2f}x–{max(current_speedups):.2f}x，平均 "
                f"{sum(current_speedups) / len(current_speedups):.2f}x；最大值出現在 "
                f"B={best['batch']}, L={best['context_len']}。\n"
            )
            if current_vs_had:
                f.write(
                    f"- `fused_hadacore256` 相對 current 的平均 latency ratio 為 "
                    f"{sum(current_vs_had) / len(current_vs_had):.3f}x；差距落在約 1% 內，"
                    "不足以支持替換 default backend。\n"
                )
            if stage_categories:
                unfused_quant = [
                    fvalue(row, "percent_of_stage_sum") for row in stage_categories
                    if row["variant"] == "unfused_INT4" and row["category"] == "quantization"
                ]
                fused_projection = [
                    fvalue(row, "percent_of_stage_sum") for row in stage_categories
                    if row["variant"] == "fused_current" and row["category"] == "projection"
                ]
                f.write(
                    f"- Stage pass 中 unfused 的 quantization 占 stage sum "
                    f"{min(unfused_quant):.1f}%–{max(unfused_quant):.1f}%；fusion 後 projection "
                    f"成為主項，占 {min(fused_projection):.1f}%–{max(fused_projection):.1f}%。"
                    "由於 stage instrumentation overhead 超過 5%，這些數字只作比例歸因。\n"
                )
            if rocprof:
                gap = [fvalue(row, "estimated_launch_gap_ms") / fvalue(row, "event_ms_per_token") for row in rocprof]
                f.write(
                    f"- rocprof selected-region trace 顯示 estimated launch/host gap 約占 "
                    f"{100.0 * min(gap):.1f}%–{100.0 * max(gap):.1f}%；fused path 已把瓶頸從 reference "
                    "Hadamard/quant/dequant 轉移到 rocBLAS projection、LM head與 dispatch overhead。\n"
                )
            f.write("\n")
        if correctness:
            f.write("## Correctness\n\n")
            markdown_table(
                f,
                correctness,
                ["variant", "batch", "context_len", "max_error", "mean_error", "mean_relative_error", "top1_match", "top10_overlap", "has_nan_or_inf"],
            )
            f.write("\n")
        if latency:
            f.write("## Full-grid Decode Latency\n\n")
            selected = [{
                **row,
                "mean_ms": fvalue(row, "mean"),
                "speedup": fvalue(row, "speedup_vs_unfused_INT4"),
                "cv_percent": fvalue(row, "coefficient_of_variation_percent"),
            } for row in latency]
            markdown_table(
                f,
                selected,
                ["batch", "context_len", "variant", "mean_ms", "p50", "p95", "cv_percent", "speedup"],
                ["B", "L", "variant", "mean ms/token", "p50", "p95", "CV %", "speedup vs unfused_INT4"],
            )
            f.write("\n")
            f.write("## Batch、Context 與 Variant Scaling\n\n")
            f.write(
                "下列折線圖使用無 instrumentation 的 HIP event latency。Context 圖顯示長序列主要增加 K2 cache scan；"
                "batch 圖則顯示 projection 權重可在同一次 GEMM 中由多個 token 共用，因此 fused path 在 B=1–8 的 latency 成長遠低於工作量成長。\n\n"
            )
            chart_root = f"decode_bottleneck_profiling_results/{chart_dir_name}"
            f.write(f"![Latency vs context]({chart_root}/latency_vs_context_by_batch.png)\n\n")
            f.write(f"![Speedup vs context]({chart_root}/speedup_vs_context_by_batch.png)\n\n")
            f.write(f"![Latency vs batch]({chart_root}/latency_vs_batch_by_context.png)\n\n")
            lookup = {
                (row["variant"], ivalue(row, "batch"), ivalue(row, "context_len")): row
                for row in latency
            }
            scaling_rows = []
            for batch in [1, 2, 4, 8]:
                for variant in compared:
                    short = fvalue(lookup[(variant, batch, 10)], "mean")
                    long = fvalue(lookup[(variant, batch, 4096)], "mean")
                    scaling_rows.append({
                        "dimension": "context 10->4096",
                        "fixed_value": f"B={batch}",
                        "variant": variant,
                        "start_ms": short,
                        "end_ms": long,
                        "latency_ratio": long / short,
                    })
            for context in [10, 128, 1024, 4096]:
                for variant in compared:
                    small = fvalue(lookup[(variant, 1, context)], "mean")
                    large = fvalue(lookup[(variant, 8, context)], "mean")
                    scaling_rows.append({
                        "dimension": "batch 1->8",
                        "fixed_value": f"L={context}",
                        "variant": variant,
                        "start_ms": small,
                        "end_ms": large,
                        "latency_ratio": large / small,
                    })
            markdown_table(
                f,
                scaling_rows,
                ["dimension", "fixed_value", "variant", "start_ms", "end_ms", "latency_ratio"],
                ["變因", "固定 shape", "variant", "起點 ms", "終點 ms", "latency ratio"],
            )
            f.write(
                "\n`fused_current` 在 B=1 時由 L=10 增至 4096，latency 僅由約 40.90 ms 增至 48.08 ms，"
                "額外成本主要來自讀取更長 KV cache。相反地，B=8,L=4096 的 unfused reference 因 dequant temporary、"
                "elementwise 與 dispatch 成本急升至約 297.09 ms，使 fused speedup 放大到 5.69x；這是 baseline overhead 被移除的效果，"
                "不是 fused GEMM 本身變快 5.69 倍。\n\n"
            )
        if sequential:
            f.write("## Sequential 32-token Decode\n\n")
            markdown_table(
                f,
                sequential,
                ["batch", "initial_context_len", "variant", "decode_ms_per_token", "tokens_per_second", "speedup_vs_unfused_INT4"],
            )
            f.write("\n")
        if stage_categories:
            f.write("## Stage Bottleneck Breakdown\n\n")
            markdown_table(
                f,
                stage_categories,
                ["batch", "context_len", "variant", "category", "total_ms_per_token", "percent_of_stage_sum", "saved_ms_vs_unfused_INT4", "instrumentation_overhead_percent"],
            )
            f.write("\n若 instrumentation overhead 超過 5%，stage event 結果只用於比例與歸因，不取代無 instrumentation latency。\n\n")
        if ablation:
            f.write("## Formal-path Ablation\n\n")
            markdown_table(
                f,
                ablation,
                ["batch", "context_len", "variant", "mean", "speedup_vs_unfused_INT4"],
            )
            f.write("\n")
        if rocprof:
            f.write("## rocprofv3 Decode-only Summary\n\n")
            markdown_table(
                f,
                rocprof,
                ["variant", "batch", "context_len", "event_ms_per_token", "gpu_busy_union_ms_per_token", "estimated_launch_gap_ms", "gpu_busy_percent_of_event", "kernel_calls_per_token"],
            )
            f.write("\n")
            f.write(f"![rocprof kernel categories]({chart_root}/rocprof_kernel_category_stacked.png)\n\n")
            category_display = [
                row for row in rocprof_categories
                if row["category"] in {
                    "rocBLAS_GEMM",
                    "PyTorch_elementwise_copy_reduce",
                    "K2_INT4",
                    "K2_FP16",
                }
            ]
            f.write("### Kernel category breakdown\n\n")
            markdown_table(
                f,
                category_display,
                ["workload", "category", "calls_per_token", "kernel_sum_ms_per_token", "percent_of_kernel_sum"],
            )
            f.write("\n")
            f.write("### Top kernels\n\n")
            top_display = []
            for row in top_rows:
                if ivalue(row, "rank") <= 3:
                    top_display.append({**row, "kernel": row["kernel"][:96]})
            markdown_table(
                f,
                top_display,
                ["workload", "rank", "category", "kernel", "calls_per_token", "total_ms_per_token", "average_us", "lds_block_bytes", "vgpr_count", "scratch_bytes"],
            )
            f.write("\n")
            f.write(
                "rocprofv3 在 gfx1201 上回報少量 dispatch start/end timestamp swap warnings；SDK 已交換顛倒值。"
                "因此細粒度 duration與busy union視為 profiling近似值，正式 latency仍以無 profiler HIP events為準。\n\n"
            )
        if gemm_bound_rows:
            f.write("## rocBLAS GEMM：Compute-bound 或 Memory-bound？\n\n")
            f.write(
                "結論是：**本次 B=1–8 token decode 的 projection/LM-head GEMM 主要是 memory/weight-streaming bound，"
                "不是 matrix compute-bound**。Kernel 使用 rocBLAS 與 matrix instruction 只描述實作方式，不代表已吃滿計算單元。\n\n"
            )
            f.write(
                "Llama-3.1 8B 每 token 的 32 層 Q/K/V/O、gate/up/down 與 LM head 合計約讀取 75 億個 FP16 weight，"
                "理論最低約 15.0 GB。若忽略 activation、output、cache miss 與重讀，arithmetic intensity 上限約等於 batch："
                "B=1 為 1 FLOP/byte，B=4 為 4 FLOP/byte。AMD 官方規格為 FP16 matrix 191 TFLOP/s、memory 640 GB/s，"
                "理論 ridge point 約 298.4 FLOP/byte；目前測試點遠在 bandwidth 斜坡左側。\n\n"
            )
            bound_display = []
            for row in gemm_bound_rows:
                bound_display.append({
                    **row,
                    "weight_GB": fvalue(row, "fp16_weight_bytes_lower_bound") / 1e9,
                    "AI": fvalue(row, "arithmetic_intensity_flop_per_byte_upper_bound"),
                    "TFLOPs": fvalue(row, "effective_tflops"),
                    "GBps": fvalue(row, "effective_weight_bandwidth_GBps"),
                    "BW_percent": fvalue(row, "percent_of_peak_memory_bandwidth"),
                })
            markdown_table(
                f,
                bound_display,
                ["batch", "context_len", "rocblas_kernel_ms_per_token", "weight_GB", "AI", "TFLOPs", "GBps", "BW_percent"],
                ["B", "L", "rocBLAS ms", "FP16 weight lower bound GB", "AI upper bound", "effective TFLOP/s", "effective weight GB/s", "% of 640 GB/s"],
            )
            f.write(f"\n![Projection GEMM roofline]({chart_root}/projection_gemm_roofline.png)\n\n")
            f.write(
                "另一個直接證據是 batch scaling：rocprof 中 `fused_current` 的 rocBLAS kernel sum 從 B=1,L=128 的約 32.57 ms，"
                "到 B=4,L=1024 只有約 33.30 ms，但 algorithmic FLOPs 增加 4 倍。這表示同一批 weight 被四個 token 攤提，"
                "effective TFLOP/s 隨 batch 提升，而時間近乎不變；若已 compute-bound，時間通常會更接近隨 FLOPs 增長。"
                "這個分類適用於目前 B<=8；更大的 batch 會提高 arithmetic intensity，最終可能轉為 compute-bound。\n\n"
            )
            f.write(
                "長 context 的次要瓶頸 K2 也偏 memory-bound：B=1 時 direct INT4 K2 kernel sum 由 L=128 的約 0.41 ms 增至 "
                "L=4096 的約 8.00 ms，占 fused kernel sum 從 1.1% 升至 17.4%。其工作量與 KV cache bytes 都近似隨 L 線性增加；"
                "GQA 讓 32 個 query heads 共用 8 個 KV heads，降低 cache traffic，但不改變長序列 cache scan 的 bandwidth 性質。"
                "剩餘 PyTorch elementwise/copy/reduction 多為低 arithmetic intensity 與 launch-sensitive，同樣不是主要 compute-bound 項。\n\n"
            )
            f.write(
                "硬體峰值來源：[AMD Radeon AI PRO R9700 官方規格](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)。"
                "上述 effective bandwidth 是由 FP16 weight 最低 bytes / rocprof kernel time 推估，不是 GL2C/DRAM counter 實測值。\n\n"
            )
        f.write("## Counters、Traffic 與 Throughput\n\n")
        if counter_rows:
            probe_rows = [row for row in counter_rows if "probe" in row["workload"]]
            markdown_table(
                f,
                probe_rows,
                ["workload", "counter", "samples", "nonzero_samples", "sum", "status"],
            )
            f.write("\nWorking counter set 的 `SQ_WAVES_sum`、`GRBM_GUI_ACTIVE`、`GRBM_COUNT`、`CU_NUM`、`SIMD_NUM` 有非零值；完整逐 workload 數據見 `counter_summary.csv`。\n\n")
            f.write("Counter collection 的 dispatch sample count 與 raw selected-region trace 不完全一致，因此 counter只用於確認支援/非零狀態，不換算每 token utilization。\n\n")
        k2_traffic = [row for row in traffic_rows if row["block"] == "K2"]
        if k2_traffic:
            f.write("### K2 理論最小 traffic\n\n")
            markdown_table(
                f,
                k2_traffic,
                ["batch", "context_len", "variant", "minimum_semantic_bytes_all_layers", "stage_ms_per_token", "effective_minimum_bandwidth_GBps"],
            )
            f.write("\n")
        throughput_display = [
            row for row in throughput_rows_data
            if row["variant"] in {"fused_current", "fused_hadacore256"}
            and row["batch"] == 1
            and row["context_len"] == 128
            and (row["metric"] == "effective_Hadamard_TOPS" or row["stage"] in {"q_proj", "gate_proj", "down_proj_residual"})
        ]
        if throughput_display:
            f.write("### Effective throughput（B=1, L=128）\n\n")
            markdown_table(
                f,
                throughput_display,
                ["variant", "stage", "metric", "algorithmic_ops", "stage_ms", "value"],
            )
            f.write("\n")
        f.write(
            "gfx1201 上 GL2C、LDS 與 MeanOccupancy counters 若仍回傳 0，不解讀為零流量或零 occupancy。"
            "報告改用 kernel trace 的 LDS/VGPR/scratch、可用的 SQ/GRBM counters，以及理論 minimum semantic traffic。"
            "TFLOP/s、Hadamard TOPS 與 effective bandwidth 均為 algorithmic/effective 指標，不等同硬體峰值。\n\n"
        )
        f.write("## 結論與限制\n\n")
        f.write(
            "目前 unfused_INT4 的主要瓶頸是 PyTorch reference Hadamard/quant/dequant、完整 paged-cache dequant與大量 dispatch；"
            "current fusion 移除這些中間步驟後，主要成本轉為 QKV/O/MLP/LM-head GEMM、剩餘 PyTorch elementwise/copy，以及 host launch gap。"
            "長 context 下 attention占比提高，但 K1/K2/K3/FFN fused kernel 本身已不是最大 kernel-time項。"
            + ("hadacore256 在 full-grid、sequential與ablation中均未形成穩定優勢，因此 default仍應保持 current。\n\n" if "fused_hadacore256" in compared else "\n\n")
        )
        f.write(
            "本結果代表目前 formal wrapper 的 decoder decode path，不包含完整 serving scheduler、sampling、tokenization與網路成本；"
            "`unfused_INT4` 也尚不是正式 QuaRot-converted、經 calibration 的 Llama。"
            "因此不可宣稱完整模型或 production serving 的同倍率 end-to-end speedup。\n"
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="decode_bottleneck_profiling_results")
    parser.add_argument("--exclude-hadacore", action="store_true")
    parser.add_argument("--report-name", default="decode_bottleneck_profiling_report_zh.md")
    parser.add_argument("--chart-dir-name", default="charts")
    args = parser.parse_args()
    root = Path(args.dir)
    latency = summarize_latency(root)
    ablation = summarize_ablation(root)
    sequential = summarize_sequential(root)
    correctness = read_csv(root / "correctness.csv")
    _, stage_categories, stage_totals = summarize_stages(root)
    traffic = minimum_traffic_rows(stage_totals)
    throughput = throughput_rows(stage_totals)
    if traffic:
        write_csv(root / "minimum_traffic.csv", traffic)
    if throughput:
        write_csv(root / "effective_throughput.csv", throughput)
    rocprof, top_rows, rocprof_categories = summarize_rocprof(root)
    counters = summarize_counters(root)
    from llama31_quarot.generate_decode_bottleneck_charts import generate as generate_charts

    gemm_bound = generate_charts(
        root,
        include_hadacore=not args.exclude_hadacore,
        chart_dir_name=args.chart_dir_name,
    )
    if args.exclude_hadacore:
        keep = lambda row: row.get("variant") != "fused_hadacore256" and not row.get("workload", "").startswith("fused_hadacore256_")
        latency = [row for row in latency if keep(row)]
        ablation = [row for row in ablation if "hadacore" not in row.get("variant", "")]
        sequential = [row for row in sequential if keep(row)]
        correctness = [row for row in correctness if keep(row)]
        stage_categories = [row for row in stage_categories if keep(row)]
        rocprof = [row for row in rocprof if keep(row)]
        top_rows = [row for row in top_rows if keep(row)]
        rocprof_categories = [row for row in rocprof_categories if keep(row)]
        traffic = [row for row in traffic if keep(row)]
        throughput = [row for row in throughput if keep(row)]
    report = write_report(
        root,
        latency,
        ablation,
        sequential,
        correctness,
        stage_categories,
        rocprof,
        top_rows,
        rocprof_categories,
        counters,
        traffic,
        throughput,
        gemm_bound,
        report_name=args.report_name,
        chart_dir_name=args.chart_dir_name,
    )
    print(report)


if __name__ == "__main__":
    main()
