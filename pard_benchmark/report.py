from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_results(directory: Path) -> list[dict]:
    results = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "environment.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("schema_version") == 1 and "repeats" in data:
            data["_path"] = path.name
            results.append(data)
    return results


def summarize(directory: Path) -> dict:
    results = load_results(directory)
    latency_rows: list[dict] = []
    memory_rows: list[dict] = []
    summary_rows: list[dict] = []
    parity_rows: list[dict] = []
    baselines: dict[tuple, list[int]] = {}
    for result in results:
        key = (result["phase"], result["target_key"], result["context_len"], result["max_new_tokens"], result["compile_mode"])
        if result["mode"] == "ar" and result["repeats"]:
            baselines[key] = result["repeats"][0]["output_ids"]
    for result in results:
        common = {
            "source": result["_path"],
            "phase": result["phase"],
            "target": result["target_key"],
            "mode": result["mode"],
            "draft_k": result["draft_k"],
            "context_len": result["context_len"],
            "max_new_tokens": result["max_new_tokens"],
            "compile_mode": result["compile_mode"],
        }
        for repeat in result["repeats"]:
            row = {**common, "repeat": repeat["repeat"]}
            for name, value in repeat.items():
                if name not in {"repeat", "output_ids", "memory"}:
                    row[name] = value
            latency_rows.append(row)
            for snap in repeat["memory"]:
                memory_rows.append({**common, "repeat": repeat["repeat"], **snap})
        for snap in result.get("load_memory", []):
            memory_rows.append({**common, "repeat": -1, **snap})
        speeds = [x["steady_tokens_per_s"] for x in result["repeats"]]
        e2e = [x["end_to_end_tokens_per_s"] for x in result["repeats"]]
        peak = max(
            (s["peak_reserved_bytes"] for x in result["repeats"] for s in x["memory"]),
            default=0,
        )
        free = min(
            (s["gpu_free_bytes"] for x in result["repeats"] for s in x["memory"]),
            default=0,
        )
        summary_rows.append(
            {
                **common,
                "steady_tokens_per_s_median": statistics.median(speeds),
                "steady_tokens_per_s_mean": statistics.mean(speeds),
                "end_to_end_tokens_per_s_median": statistics.median(e2e),
                "peak_reserved_bytes": peak,
                "minimum_gpu_free_bytes": free,
                "target_parameter_bytes": result["parameter_bytes"]["target"],
                "draft_parameter_bytes": result["parameter_bytes"]["draft"],
                "gpu_total_bytes": result["environment"]["device_total_memory"],
            }
        )
        baseline = baselines.get((result["phase"], result["target_key"], result["context_len"], result["max_new_tokens"], result["compile_mode"]))
        candidate = result["repeats"][0]["output_ids"] if result["repeats"] else []
        parity_rows.append(
            {
                **common,
                "baseline_available": baseline is not None,
                "exact_token_parity": baseline == candidate if baseline is not None else result["mode"] == "ar",
                "first_mismatch": first_mismatch(baseline, candidate) if baseline is not None else "",
            }
        )
    baselines_speed = {
        (x["phase"], x["target"], x["context_len"], x["max_new_tokens"], x["compile_mode"]): x["steady_tokens_per_s_median"]
        for x in summary_rows
        if x["mode"] == "ar"
    }
    for row in summary_rows:
        base = baselines_speed.get((row["phase"], row["target"], row["context_len"], row["max_new_tokens"], row["compile_mode"]))
        row["speedup_vs_ar"] = row["steady_tokens_per_s_median"] / base if base else ""
    write_csv(directory / "latency_repeats.csv", latency_rows)
    write_csv(directory / "latency_summary.csv", summary_rows)
    write_csv(directory / "memory.csv", memory_rows)
    write_csv(directory / "generation_check.csv", parity_rows)
    if results:
        (directory / "environment.json").write_text(
            json.dumps(
                {
                    "upstream_commit": results[0]["upstream_commit"],
                    "models": {
                        **{f"target:{x['target_key']}": x["target"] for x in results},
                        **{
                            f"draft:{x['mode']}": x["draft"]
                            for x in results
                            if x.get("draft") is not None
                        },
                    },
                    "environments": {x["compile_mode"]: x["environment"] for x in results},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    fallback_path = directory / "compile_fallbacks.json"
    compile_fallback = json.loads(fallback_path.read_text(encoding="utf-8")) if fallback_path.exists() else None
    report = render_report(summary_rows, parity_rows, compile_fallback)
    (directory / "report_zh.md").write_text(report, encoding="utf-8")
    return {"results": len(results), "summary_rows": len(summary_rows), "report": str(directory / "report_zh.md")}


def first_mismatch(left, right):
    if left is None:
        return ""
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    return "" if len(left) == len(right) else min(len(left), len(right))


def render_report(summary_rows: list[dict], parity_rows: list[dict], compile_fallback: dict | None = None) -> str:
    parity = {(x["source"]): x["exact_token_parity"] for x in parity_rows}
    formal_rows = [x for x in summary_rows if x["phase"] == "formal"]
    displayed_rows = formal_rows or [x for x in summary_rows if x["phase"] == "smoke"]
    lines = [
        "# PARD / PARD2 Llama 3.1 Decode 評估報告",
        "",
        "> 本報告只比較相同 target、dtype、context 與 compile mode；不與既有 FP16 QuaRot 歷史結果直接計算 speedup。",
        "",
        "## 正式結果" if formal_rows else "## Smoke 結果",
        "",
        "| target | mode | k | context | compile | steady tok/s | speedup vs AR | token parity | peak reserved GiB | min free GiB |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in sorted(displayed_rows, key=lambda x: (x["target"], x["compile_mode"], x["context_len"], x["mode"], x["draft_k"])):
        speedup = row["speedup_vs_ar"]
        speedup_text = f"{speedup:.3f}" if isinstance(speedup, float) else "n/a"
        k_text = "-" if row["mode"] == "ar" else str(row["draft_k"])
        lines.append(
            f"| {row['target']} | {row['mode']} | {k_text} | {row['context_len']} | "
            f"{row['compile_mode']} | {row['steady_tokens_per_s_median']:.3f} | {speedup_text} | "
            f"{'PASS' if parity.get(row['source']) else 'FAIL/n/a'} | "
            f"{row['peak_reserved_bytes'] / 2**30:.3f} | {row['minimum_gpu_free_bytes'] / 2**30:.3f} |"
        )
    lines.extend(["", "## 採用判定", ""])
    grouped = defaultdict(list)
    for row in formal_rows:
        if row["mode"] != "ar" and isinstance(row["speedup_vs_ar"], float):
            grouped[(row["target"], row["mode"], row["draft_k"], row["compile_mode"])].append(row)
    if not grouped:
        lines.append("尚無完整、可與 AR 配對的正式結果。")
    for key, rows in sorted(grouped.items()):
        faster = sum(x["speedup_vs_ar"] > 1.0 for x in rows)
        parity_ok = all(parity.get(x["source"], False) for x in rows)
        headroom_ok = all(x["minimum_gpu_free_bytes"] >= 0.10 * x["gpu_total_bytes"] for x in rows)
        accepted = parity_ok and faster >= 2 and headroom_ok
        lines.append(
            f"- `{key[0]} / {key[1]} / k={key[2]} / {key[3]}`："
            f"parity={'PASS' if parity_ok else 'FAIL'}，較快 context={faster}/{len(rows)}，"
            f"VRAM headroom={'PASS' if headroom_ok else 'FAIL'}，結論={'建議採用' if accepted else '暫不採用'}。"
        )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 這是獨立 Transformers+ PARD benchmark，尚未接入 QuaRot multi-token verifier 或 KV rollback。",
            "- 若 token parity 失敗，該列速度只供除錯，不可視為 lossless speculative decoding speedup。",
            "- GPU-wide free memory 會受其他程序影響；正式測試應在 preflight 無外部 VRAM 佔用時執行。",
            "",
        ]
    )
    if compile_fallback:
        lines.insert(
            4,
            f"> compile：原先要求 `{compile_fallback['requested_compile_mode']}`，實測改為整組 "
            f"`{compile_fallback['effective_compile_mode']}`；未將未完成的 compile case 混入比較。",
        )
        lines.insert(5, "")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="pard_decode_results")
    args = parser.parse_args()
    print(json.dumps(summarize(Path(args.dir))))


if __name__ == "__main__":
    main()
