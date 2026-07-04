#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "single_decoder_layer_results"
ROCPROF = ROOT / "rocprof_single_decoder_layer"
REPORT = ROOT / "single_decoder_layer_report_zh.md"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(x, nd=3):
    return f"{float(x):.{nd}f}"


def table(f, rows, cols):
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for row in rows:
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.3f}"
            vals.append(str(val))
        f.write("| " + " | ".join(vals) + " |\n")


def latency_summary(latency):
    grouped = defaultdict(dict)
    for row in latency:
        key = (row["batch"], row["seq_len"], row["ffn_hidden"])
        grouped[key][row["variant"]] = float(row["latency_ms"])
    rows = []
    for (batch, seq_len, ffn_hidden), vals in sorted(grouped.items(), key=lambda x: tuple(map(int, x[0]))):
        rows.append({
            "batch": batch,
            "seq_len": seq_len,
            "ffn_hidden": ffn_hidden,
            "fp16_ms": vals["fp16_baseline"],
            "unfused_ms": vals["quarot_unfused"],
            "fused_ms": vals["fused_quarot"],
            "fused_vs_unfused": vals["quarot_unfused"] / vals["fused_quarot"],
            "fused_vs_fp16": vals["fp16_baseline"] / vals["fused_quarot"],
        })
    return rows


def correctness_summary(correctness):
    rows = []
    for row in correctness:
        if row["variant"] == "fused_quarot" and row["reference"] == "quarot_unfused":
            rows.append({
                "batch": row["batch"],
                "seq_len": row["seq_len"],
                "ffn_hidden": row["ffn_hidden"],
                "max_error": float(row["max_error"]),
                "mean_error": float(row["mean_error"]),
                "mean_relative_error": float(row["mean_relative_error"]),
                "tolerance": row.get("tolerance", ""),
                "result": row.get("result", ""),
            })
    return sorted(rows, key=lambda r: (int(r["batch"]), int(r["seq_len"]), int(r["ffn_hidden"])))


def aggregate_ranges(lat_summary):
    unfused_speedups = [r["fused_vs_unfused"] for r in lat_summary]
    fp16_speedups = [r["fused_vs_fp16"] for r in lat_summary]
    return {
        "fused_vs_unfused_min": min(unfused_speedups),
        "fused_vs_unfused_mean": sum(unfused_speedups) / len(unfused_speedups),
        "fused_vs_unfused_max": max(unfused_speedups),
        "fused_vs_fp16_min": min(fp16_speedups),
        "fused_vs_fp16_mean": sum(fp16_speedups) / len(fp16_speedups),
        "fused_vs_fp16_max": max(fp16_speedups),
    }


def representative(rows, batch="1", ffn_hidden="14336"):
    return [r for r in rows if r["batch"] == batch and r["ffn_hidden"] == ffn_hidden]


def theoretical_memory_rows():
    b, l, d, h = 1, 128, 4096, 14336
    return [
        {
            "component": "K2 FP16 KV cache read",
            "formula": "B * L * D * 2(K,V) * 2 bytes",
            "bytes": b * l * d * 2 * 2,
        },
        {
            "component": "K2 INT4 KV cache read",
            "formula": "B * L * D * 2(K,V) * 0.5 bytes + scales",
            "bytes": b * l * d * 2 // 2 + b * l * 32 * 2 * 2 * 2,
        },
        {
            "component": "K1 current read/write",
            "formula": "read FP16 K/V, write INT4 K/V + scale",
            "bytes": b * d * 2 * 2 + b * d + b * 32 * 2 * 2 * 2,
        },
        {
            "component": "K3 output quant",
            "formula": "read FP16 4096, write INT4 4096 + 16 scales",
            "bytes": b * d * 2 + b * d // 2 + b * 16 * 2,
        },
        {
            "component": "FFN intermediate quant",
            "formula": "read FP16 gate/up, write INT4 hidden + scales",
            "bytes": b * h * 2 * 2 + b * h // 2 + b * (h // 256) * 2,
        },
    ]


def main():
    latency = read_csv(RESULTS / "latency.csv")
    correctness = read_csv(RESULTS / "correctness.csv")
    ablation = read_csv(RESULTS / "ablation.csv")
    rocprof_text = (ROCPROF / "summary.md").read_text() if (ROCPROF / "summary.md").exists() else ""

    lat_summary = latency_summary(latency)
    corr_summary = correctness_summary(correctness)
    agg = aggregate_ranges(lat_summary)

    with REPORT.open("w") as f:
        f.write("# Single Decoder Layer / QuaRot Fusion 評估報告\n\n")
        f.write("## 實驗環境\n\n")
        f.write("- 日期：2026-07-04\n")
        f.write("- 容器：`rocm/vllm-dev:rocm7.2_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.14.0rc0`\n")
        f.write("- GPU/ROCm：使用本機 ROCm HIP runtime；rocprof-compute 目前不支援 `gfx1201`，rocprofv3 可收 kernel trace/stats。\n")
        f.write("- 程式：`single_decoder_layer_benchmark.py`，輸出在 `single_decoder_layer_results/`，rocprof 在 `rocprof_single_decoder_layer/`。\n\n")

        f.write("## Benchmark 設定\n\n")
        f.write("- Decoder layer decode-step harness：前 `L-1` token KV cache 已 prefilled；timed region 包含 current token RMSNorm、QKV projection、RoPE/KV append、attention decode、O projection/residual、FFN gate/up/down projection。\n")
        f.write("- 三條主路徑：`fp16_baseline`、`quarot_unfused`、`fused_quarot`。\n")
        f.write("- Ablation：`quarot_unfused`、`k1_fused`、`k1_k2_fused`、`attention_fused`、`fused_quarot`。\n")
        f.write("- Shapes：batch = 1,2,4,8；context length L = 10,128,1024,4096；FFN hidden = 11008,14336。\n")
        f.write("- Timing：HIP/PyTorch CUDA event，`warmup=2`、`iters=5`。\n")
        f.write("- 權重/輸入：固定 random seed；三條路徑共用同一組 current hidden、cache state 與 layer weights。\n\n")

        f.write("## Correctness\n\n")
        f.write("下表為 fused QuaRot output hidden states 相對 QuaRot original unfused 的差異。relative error 會被接近 0 的元素放大，因此判讀時以 max/mean error 為主。\n\n")
        table(f, representative(corr_summary), ["batch", "seq_len", "ffn_hidden", "max_error", "mean_error", "mean_relative_error", "tolerance", "result"])
        f.write("\n完整 correctness CSV：`single_decoder_layer_results/correctness.csv`。\n\n")

        f.write("## Latency\n\n")
        f.write(
            f"全 shape 平均：fused QuaRot 相對 QuaRot unfused speedup = "
            f"{agg['fused_vs_unfused_mean']:.2f}x，範圍 {agg['fused_vs_unfused_min']:.2f}x 到 {agg['fused_vs_unfused_max']:.2f}x；"
            f"相對 FP16 baseline 平均 = {agg['fused_vs_fp16_mean']:.2f}x，範圍 {agg['fused_vs_fp16_min']:.2f}x 到 {agg['fused_vs_fp16_max']:.2f}x。\n\n"
        )
        f.write("代表表：batch=1、FFN hidden=14336。\n\n")
        table(f, representative(lat_summary), ["batch", "seq_len", "ffn_hidden", "fp16_ms", "unfused_ms", "fused_ms", "fused_vs_unfused", "fused_vs_fp16"])
        f.write("\n完整 latency CSV：`single_decoder_layer_results/latency.csv`。\n\n")

        f.write("## Ablation\n\n")
        f.write("代表表：batch=1、FFN hidden=14336。\n\n")
        abl_rows = []
        for row in ablation:
            if row["batch"] == "1" and row["ffn_hidden"] == "14336":
                abl_rows.append({
                    "seq_len": row["seq_len"],
                    "variant": row["variant"],
                    "latency_ms": float(row["latency_ms"]),
                    "speedup_vs_unfused": float(row["speedup_vs_unfused"]),
                })
        table(f, abl_rows, ["seq_len", "variant", "latency_ms", "speedup_vs_unfused"])
        f.write("\n完整 ablation CSV：`single_decoder_layer_results/ablation.csv`。\n\n")

        f.write("## rocprof Kernel Breakdown\n\n")
        f.write("rocprofv3 代表點：batch=1、L=128、FFN hidden=14336、iters=5。\n")
        f.write("rocprofv3 執行期間出現 timestamp swap warnings；CSV stats 已產生，但細粒度 kernel duration 仍應視為 profiling 近似值。\n\n")
        if rocprof_text:
            keep = []
            for line in rocprof_text.splitlines():
                if line.startswith("#") or line.startswith("`") or line.startswith("|") or line.startswith("## Top kernels"):
                    keep.append(line)
            f.write("\n".join(keep))
            f.write("\n\n")
        f.write("## Memory / Counter 補充\n\n")
        f.write("gfx1201 上 GL2C/LDS/occupancy counter 先前量測仍為 0；rocprof-compute 也回報不支援此 arch。因此本報告用 rocprofv3 kernel time/calls 搭配理論最小 memory traffic 補充。\n\n")
        mem_rows = theoretical_memory_rows()
        table(f, mem_rows, ["component", "formula", "bytes"])
        f.write("\n以上 bytes 為代表點 B=1、L=128、FFN hidden=14336 的最低資料量估算，未包含 GEMM 讀權重、cache miss、alignment、temporary tensor、PyTorch dispatcher overhead。\n\n")

        f.write("## 結論與限制\n\n")
        f.write("- 這次結果比原本只測 PyTorch reference pipeline 的 kernel-fusion ablation 更可信，因為 harness 包含 norm、projection、residual、attention decode、O/FFN projection，且只把 K1/K2/K3/FFN 替換成 fused kernel。\n")
        f.write("- Fused QuaRot 對 QuaRot original unfused path 有穩定 decoder-layer speedup，主要來自減少 separate Hadamard/quant/dequant 的 PyTorch kernel calls。\n")
        f.write("- 相對 FP16 baseline 的改善有限且 shape-dependent，因為完整 decoder layer 中 GEMM/projection 仍佔主要時間；因此不能宣稱 full LLM end-to-end 有 100x speedup。\n")
        f.write("- 目前仍是 synthetic single decoder layer harness，使用 random weights/cache；不是完整 HuggingFace/Llama model class，也未量測 tokenizer、sampling、多層模型排程與真實 logits quality。\n")
        f.write("- correctness 目前固定輸出 max/mean/relative error；fused K1 pack 與 unfused reference 允許少量 INT4 byte mismatch，最終 hidden mean error 需依專題可接受 tolerance 再定案。\n")


if __name__ == "__main__":
    main()
