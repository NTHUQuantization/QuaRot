#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def table(f, rows, cols):
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            try:
                val = f"{float(val):.4g}"
            except (ValueError, TypeError):
                pass
            vals.append(str(val))
        f.write("| " + " | ".join(vals) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="llama31_full_model_results")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.results_dir)
    out = Path(args.out) if args.out else root / "full_model_report_zh.md"
    latency = read_csv(root / "latency_summary.csv")
    quality = read_csv(root / "quality_logits.csv")
    shape_path = root / "model_shape.json"
    shape = json.loads(shape_path.read_text()) if shape_path.exists() else {}

    with out.open("w") as f:
        f.write("# Llama-3.1 8B Full Model / QuaRot Fusion 評估報告\n\n")
        f.write("## Model / Integration Status\n\n")
        if shape:
            f.write(f"- model_id: `{shape.get('model_id')}`\n")
            f.write(f"- hidden_size: `{shape.get('hidden_size')}`\n")
            f.write(f"- intermediate_size: `{shape.get('intermediate_size')}`\n")
            f.write(f"- layers: `{shape.get('num_hidden_layers')}`\n")
            f.write(f"- q_heads: `{shape.get('num_attention_heads')}`\n")
            f.write(f"- kv_heads: `{shape.get('num_key_value_heads')}`\n")
            f.write(f"- head_dim: `{shape.get('head_dim')}`\n")
            f.write(f"- current fused full-model compatibility: `{shape.get('compatible')}`\n")
            f.write(f"- note: {shape.get('reason')}\n\n")
        f.write("## Latency Summary\n\n")
        if latency:
            table(f, latency, ["mode", "batch", "context_len", "metric", "mean", "median", "std", "p90", "p95"])
        else:
            f.write("No latency summary found.\n")
        f.write("\n## Quality / Logits Summary\n\n")
        if quality:
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
        else:
            f.write("No quality summary found.\n")
        f.write("\n## Limitations\n\n")
        f.write("- Llama-3.1 8B uses GQA (`q_heads=32`, `kv_heads=8`). Current K1/K2 prototypes assume identical Q/KV head count, so fused full-model attention must wait for GQA-aware kernels.\n")
        f.write("- `fp16_hf` is the first real-model baseline. QuaRot unfused/fused full-model modes intentionally fail fast until the GQA kernel gap is closed.\n")
        f.write("- This report should replace synthetic single-layer numbers only after full-model QuaRot modes are implemented and rerun.\n")


if __name__ == "__main__":
    main()

