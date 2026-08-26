# PARD2 post-merge qualification results

本目錄是 `fused_v1 × PARD2` 合併後正式量測的索引。Raw JSON 含完整 prompt/token traces，受 repo 的 `*.json` 規則排除；可提交且足以重算 gate 的摘要位於 [`../pard2_post_merge_qualification/qualification_phase1.json`](../pard2_post_merge_qualification/qualification_phase1.json)。兩篇教程與 [`../e2e/FUSED_V1_MERGE_CONFLICTS_ZH.md`](../e2e/FUSED_V1_MERGE_CONFLICTS_ZH.md) 說明架構、實驗脈絡與衝突取捨。

## Benchmark contract

- Qwen3-8B W4A4KV4 target；batch 1、greedy、正常 EOS、最多 256 generated tokens；
- HumanEval 80、GSM8K 80、MATH-500 20 prompts；
- 每個 mode/dataset 獨立程序，8 個非計分 warmups、3 個 measured sweeps；
- drafter 使用 `max-autotune-no-cudagraphs` 並在量測前 materialize M=15–30 proposal shapes；
- speedup 逐 `(sweep, prompt)` 與同輪 post-merge AR 配對，再取 median 與 10,000-sample bootstrap CI。

## Aggregate results

| Dataset | AR steady tok/s | TI steady tok/s | TI paired speedup | TI accept | TD steady tok/s | TD paired speedup | TD accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| HumanEval | 30.967 | 49.783 | 1.607× | 5.656 | 57.438 | 1.854× | 6.431 |
| GSM8K | 31.036 | 49.785 | 1.602× | 5.548 | 59.542 | 1.910× | 6.355 |
| MATH-500 | 30.005 | 52.688 | 1.789× | 5.787 | 60.209 | 2.015× | 6.489 |

TI/TD peak VRAM 為 8.20–8.70 GiB；最差 headroom 仍為 72.7%。六組 exact parity，CI 下界皆大於 1，steady CV 皆低於 5%。MATH TD 第一次 E2E CV 受首載影響，cache-warm rerun 降為 1.83%，steady TPS 只差 −0.085%；qualification 使用 rerun，第一次 raw file保留供稽核。

## Raw artifact names

本機完整結果預期使用以下名稱：

```text
ar_humaneval.json
ti_humaneval_nocg_precompiled.json
td_humaneval_nocg_precompiled.json
ar_gsm8k.json
ti_gsm8k_nocg_precompiled.json
td_gsm8k_nocg_precompiled.json
ar_math_500.json
ti_math_500_nocg_precompiled.json
td_math_500_nocg_precompiled_rerun.json
```

## Reproduce one run

```bash
python e2e/benchmark_pard2.py \
  --mode pard2-td \
  --dataset math_500 \
  --generated-tokens 256 \
  --warmups 8 \
  --sweeps 3 \
  --compile-mode max-autotune-no-cudagraphs \
  --precompile-draft-shapes \
  --output /tmp/td_math_500.json
```

正式比較應依 AR、TI、TD 各自開新程序並保存九份 JSON；不要用 smoke 的 `--limit/--offset` 結果作採用結論。完整測試結果為 `239 passed`，HIP extension clean-build export qualification 為 26/26。
