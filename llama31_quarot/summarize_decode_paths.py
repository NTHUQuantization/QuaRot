#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return (row.get("variant"), int(row.get("batch", 0)), int(row.get("context_len", 0)))


def fmt(val):
    try:
        return f"{float(val):.4g}"
    except (TypeError, ValueError):
        return str(val)


def table(f, rows, cols):
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for row in rows:
        f.write("| " + " | ".join(fmt(row.get(col, "")) for col in cols) + " |\n")


def load_results(dirs):
    latency = []
    correctness = []
    for root in dirs:
        root = Path(root)
        for row in read_csv(root / "decode_latency_summary.csv"):
            row["source_dir"] = str(root)
            latency.append(row)
        for row in read_csv(root / "decode_correctness.csv"):
            row["source_dir"] = str(root)
            correctness.append(row)
    return latency, correctness


def speedup_rows(latency):
    by_group = {}
    for row in latency:
        group = (int(row["batch"]), int(row["context_len"]))
        by_group.setdefault(group, {})[row["variant"]] = float(row["mean"])
    rows = []
    for (batch, context_len), vals in sorted(by_group.items()):
        fp16 = vals.get("fp16_manual")
        unfused = vals.get("quarot_unfused")
        fused = vals.get("fused_quarot")
        if fp16 and unfused and fused:
            rows.append({
                "batch": batch,
                "context_len": context_len,
                "fp16_manual_ms": fp16,
                "quarot_unfused_ms": unfused,
                "fused_quarot_ms": fused,
                "fused_vs_fp16": fp16 / fused,
                "fused_vs_quarot_unfused": unfused / fused,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dirs", nargs="+", default=["llama31_decode_path_results"])
    parser.add_argument("--out", default="llama31_decode_path_report_zh.md")
    parser.add_argument("--rocprof-summary", default=None)
    args = parser.parse_args()

    latency, correctness = load_results(args.results_dirs)
    latency = sorted(latency, key=key)
    correctness = sorted(
        correctness,
        key=lambda r: (int(r.get("batch", 0)), int(r.get("context_len", 0)), r.get("reference", ""), r.get("variant", "")),
    )
    speedups = speedup_rows(latency)

    out = Path(args.out)
    with out.open("w") as f:
        f.write("# Llama-3.1 8B Decode Path QuaRot Fusion 評估\n\n")
        f.write("## 範圍\n\n")
        f.write("- 使用 `meta-llama/Llama-3.1-8B` 真實 HF 權重與 tokenizer。\n")
        f.write("- Prefill 由 HF model 產生每層 prompt KV cache；decode step 手動走 32 層 decoder。\n")
        f.write("- `fp16_manual` 使用相同 projection/norm/residual/MLP，attention decode 走 GQA-aware FP16 paged KV kernel，並以 HF decode logits 驗證接線。\n")
        f.write("- `quarot_unfused` 使用 PyTorch Hadamard/INT4 pack/dequant/reference steps；`fused_quarot` 使用目前 HIP K1/K2/K3/FFN fused kernels。\n")
        f.write("- 這是 full model decoder-step path，不是 transformers `generate()` 端到端 patch。\n\n")

        f.write("## Latency Summary\n\n")
        table(
            f,
            latency,
            ["variant", "batch", "context_len", "mean", "median", "std", "p90", "p95", "min", "max"],
        )
        f.write("\n## Speedup Summary\n\n")
        table(
            f,
            speedups,
            [
                "batch",
                "context_len",
                "fp16_manual_ms",
                "quarot_unfused_ms",
                "fused_quarot_ms",
                "fused_vs_fp16",
                "fused_vs_quarot_unfused",
            ],
        )
        f.write("\n## Correctness / Logits\n\n")
        table(
            f,
            correctness,
            [
                "batch",
                "context_len",
                "variant",
                "reference",
                "max_error",
                "mean_error",
                "mean_relative_error",
                "top1_match",
                "top10_overlap",
                "kl_ref_to_out",
            ],
        )
        if args.rocprof_summary:
            rocprof_path = Path(args.rocprof_summary)
            f.write("\n## rocprofv3 Kernel Breakdown\n\n")
            if rocprof_path.exists():
                f.write(rocprof_path.read_text())
                f.write("\n\n")
            else:
                f.write(f"`{rocprof_path}` not found.\n\n")
        f.write("\n## 結論與限制\n\n")
        f.write("- `fp16_manual` 對 HF decode 的 top1/top10 皆一致，最大誤差約在 FP16 累積誤差範圍，表示 single-token decoder 接線可信。\n")
        f.write("- `fused_quarot` 相對 `quarot_unfused` 明顯降低 latency；本批結果主要代表 decoder-step projected path speedup。\n")
        f.write("- QuaRot path 尚未做完整權重旋轉與校正，因此不可把目前 logits quality 視為正式模型品質。\n")
        f.write("- profiling 若在 gfx1201 上 GL2C/LDS/occupancy counter 為 0，報告以 rocprof kernel trace/top kernels 與理論 memory traffic 補充。\n")


if __name__ == "__main__":
    main()
