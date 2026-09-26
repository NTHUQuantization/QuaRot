# GPTQ A–E 工作目錄

正式結果統一見 ../../benchmark_results/cumulative_qwen3_8b/README.md。此處只放執行程式、進度與尚未驗證的工作資料。

run_queue.py 先等待 8B／14B／32B GPTQ 稽核完成，然後 pilot D/A/B/C/E → formal D/A/B/C/E → 驗證 → 整理 → 各 remote branch 推送。量化前置進度見 ../../gptq_preflight_20260925/status.json。

目前命名版本 3：A HIP（來源 main）、B Hadacore、C GEMM、D Fusion、E PARD2。已完成 pilot 重新標示後，原時間順序是 D/A/C/B/E；正式尚未跑的 B/C 按新順序接續。
