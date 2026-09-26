> 歷史紀錄說明（2026-09-26）：本目錄只證明歷史 schema 2→3 的純命名交換；不代表現行來源。2026-09-26 schema 4 又更換 B/C commit，詳見 ../source_migration_20260926/README.md。

# 命名版本 3：HIP → Hadacore → GEMM → Fusion → PARD2

- 本次五組代號修正：原 B（GEMM）→C，原 C（Hadacore）→B；A 名稱 main→HIP；D 名稱 Fused AR→Fusion。
- source branch 仍為 main/Hadacore/GEMM/fused_pard2，source commit 與 kernel 不變；raw 數值不變。
- 此修正不同於最早四組的 C→D、D→E；舊四組 raw 只作封存證據，整理後副本已用 D/E。
- original_label_v2_records.tar.gz 保存本次改名前原始檔；data_integrity.json 逐筆驗證改標未更動測量 payload。
- verify_label_migration.py 使用 AST 比較，確認 runtime_adapter 只交换 B/C branch 對應，計時 worker 完全未變；允許既有 D/A 及 pilot hash 與新 B/C/E 同時追溯。
- 實際 GPTQ pilot 改標後的時間順序為 D/A/C/B/E；報表按 A/B/C/D/E 呈現。正式剩餘 B/C/E 按新順序接續；A 不重跑。
