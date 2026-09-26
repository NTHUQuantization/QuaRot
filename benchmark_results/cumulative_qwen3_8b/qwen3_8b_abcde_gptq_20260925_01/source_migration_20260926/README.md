# 2026-09-26 來源更新（schema 4）

現行 B=GEMM（僅 GEMM 優化）、C=Hadacore（GEMM+HadaCore）。本次更換 commit，不能只重標舊資料。

`previous_schema3/` 是舊文件／腳本；`superseded_pilot/` 與 tar.gz 是舊版 B=Hadacore、C=GEMM 的 GPTQ pilot 原始資料及舊彙整，僅供追查，不屬於現行比較。A/D/E 的來源與數據保持不變；B/C 須先重跑 pilot 再正式測量。
