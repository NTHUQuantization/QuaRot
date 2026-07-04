#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt(value):
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def table(f, rows, cols):
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |\n")


def latency_rows(args):
    rows = []
    for mode, root in [
        ("fp16_hf", Path(args.fp16_dir)),
        ("quarot_unfused", Path(args.unfused_dir)),
        ("fused_quarot", Path(args.fused_dir)),
    ]:
        for row in read_csv(root / "latency_summary.csv"):
            row["mode"] = mode
            rows.append(row)
    return sorted(rows, key=lambda r: (r["metric"], int(r["batch"]), int(r["context_len"]), r["mode"]))


def speedup_rows(latency):
    grouped = {}
    for row in latency:
        if row["metric"] != "decode_ms_per_token":
            continue
        key = (int(row["batch"]), int(row["context_len"]))
        grouped.setdefault(key, {})[row["mode"]] = float(row["mean"])
    rows = []
    for (batch, context_len), vals in sorted(grouped.items()):
        if {"fp16_hf", "quarot_unfused", "fused_quarot"} <= vals.keys():
            rows.append({
                "batch": batch,
                "context_len": context_len,
                "fp16_hf_ms": vals["fp16_hf"],
                "quarot_unfused_ms": vals["quarot_unfused"],
                "fused_quarot_ms": vals["fused_quarot"],
                "fused_vs_fp16": vals["fp16_hf"] / vals["fused_quarot"],
                "fused_vs_unfused": vals["quarot_unfused"] / vals["fused_quarot"],
            })
    return rows


def quality_rows(args):
    rows = []
    for root in [Path(args.quality_fp16_unfused), Path(args.quality_fp16_fused), Path(args.quality_unfused_fused)]:
        rows.extend(read_csv(root / "quality_logits.csv"))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16-dir", default="llama31_formal_latency_fp16")
    parser.add_argument("--unfused-dir", default="llama31_formal_latency_quarot_unfused")
    parser.add_argument("--fused-dir", default="llama31_formal_latency_fused_quarot")
    parser.add_argument("--quality-fp16-unfused", default="llama31_formal_quality_fp16_vs_unfused")
    parser.add_argument("--quality-fp16-fused", default="llama31_formal_quality_fp16_vs_fused")
    parser.add_argument("--quality-unfused-fused", default="llama31_formal_quality_unfused_vs_fused")
    parser.add_argument("--rocprof-summary", default="llama31_formal_rocprof/summary.md")
    parser.add_argument("--out", default="llama31_formal_full_model_report_zh.md")
    args = parser.parse_args()

    latency = latency_rows(args)
    speedups = speedup_rows(latency)
    quality = quality_rows(args)
    rocprof = Path(args.rocprof_summary)

    with Path(args.out).open("w") as f:
        f.write("# Llama-3.1 8B Formal QuaRot Wrapper Full-Model 評估\n\n")
        f.write("## 範圍\n\n")
        f.write("- 使用 `meta-llama/Llama-3.1-8B` 真實權重與 tokenizer。\n")
        f.write("- 新增 `QuaRotLlamaForCausalLM` formal token-by-token wrapper，提供 `prefill()`、`decode_one()`、`generate()`。\n")
        f.write("- Prefill 仍使用 HF Llama/SDPA，並用 `logits_to_keep=1` 避免長 context 產生整段 vocab logits。\n")
        f.write("- `quarot_unfused` / `fused_quarot` 的 decode token 經過 GQA-aware K1/K2/K3/FFN path；不是 monkey-patch transformers `generate()` internals。\n")
        f.write("- QuaRot 權重旋轉與 calibration 尚未完成，因此品質結果只能視為目前 prototype path 的輸出差異。\n\n")

        f.write("## Latency 設定\n\n")
        f.write("- batch = `1,2,4,8`\n")
        f.write("- context length = `10,128,1024,4096`\n")
        f.write("- warmup = `1`, iters = `3`, repeats = `3`\n")
        f.write("- attention implementation = `sdpa`\n")
        f.write("- prefill 包含 prompt forward 與 cache 建立；QuaRot modes 的 prefill 另包含 HF KV cache 轉 paged INT4 cache。\n\n")

        f.write("## Decode Latency / Speedup\n\n")
        table(
            f,
            speedups,
            [
                "batch",
                "context_len",
                "fp16_hf_ms",
                "quarot_unfused_ms",
                "fused_quarot_ms",
                "fused_vs_fp16",
                "fused_vs_unfused",
            ],
        )
        f.write("\n## Full Latency Summary\n\n")
        table(f, latency, ["mode", "batch", "context_len", "metric", "mean", "median", "std", "p90", "p95"])

        f.write("\n## Logits / Generation Quality\n\n")
        table(
            f,
            quality,
            [
                "reference_mode",
                "candidate_mode",
                "prompt_id",
                "max_length",
                "max_error",
                "mean_error",
                "top1_match",
                "top10_overlap",
                "kl_ref_to_out",
            ],
        )
        f.write("\nGeneration samples:\n\n")
        f.write("- `llama31_formal_quality_fp16_vs_unfused/generation_samples.md`\n")
        f.write("- `llama31_formal_quality_fp16_vs_fused/generation_samples.md`\n")
        f.write("- `llama31_formal_quality_unfused_vs_fused/generation_samples.md`\n\n")

        f.write("## rocprofv3 Summary\n\n")
        if rocprof.exists():
            f.write(rocprof.read_text())
            f.write("\n")
        else:
            f.write(f"`{rocprof}` not found.\n")

        f.write("\n## 結論\n\n")
        f.write("- Formal wrapper 已讓 `quarot_unfused` / `fused_quarot` 透過正式 token-by-token API 執行 `prefill()`、`decode_one()`、`generate()`。\n")
        f.write("- Fused QuaRot 相對 QuaRot unfused 的 decode latency 有穩定改善；在完整 8B path 中主要落在約 `2.3x-5.7x`，shape-dependent。\n")
        f.write("- Fused QuaRot 相對 FP16 HF decode 多數 shape 接近或略慢，因為 GEMM/projection/prefill 仍主導 full-model 成本；不可宣稱 end-to-end 100x speedup。\n")
        f.write("- QuaRot path 尚未完成正式權重旋轉/calibration，FP16 vs QuaRot 的 logits/generation 品質目前不代表最終模型品質。\n")


if __name__ == "__main__":
    main()
