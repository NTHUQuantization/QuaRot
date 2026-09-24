# Qwen3 8B／14B／32B：HIP 架構設計與調整差異（簡要版）

> 範圍：本文件說明 fused_v1 的 QuaRot W4A4KV4 target 如何依 Qwen3 模型形狀選擇 HIP 執行路徑，以及 PARD2 所需的 profile 對齊。它不是三個模型的 accuracy 或速度排名；所有數字皆以 batch=1 的既有實驗設定為背景。

## 一句話摘要

- **8B** 是原始、較小的 Qwen3 基線：核心寬度為 4096，沿用通用 HIP dispatch。
- **14B** 將 hidden width 擴到 5120、層數增至 40；可共用 5120／1024 的既有路徑，但 FFN=17408 沒有專用 tile，走通用 fallback。其 TD 必須處理 5120-wide HadK 的 FP16 overflow。
- **32B** 與 14B 同為 hidden=5120，但有 64 層、64 個 Q heads、FFN=25600；因此新增且實測採用 5120／8192／25600 的 exact-shape dispatch，以及專用 grouped/multi-GEMM waves（4/2）。

## 1. 模型形狀：為何不能直接把 8B 的設定套用到更大模型

| 項目 | Qwen3-8B | Qwen3-14B | Qwen3-32B |
|---|---:|---:|---:|
| hidden size | 4096 | 5120 | 5120 |
| FFN intermediate size | 12288 | 17408 | 25600 |
| layers | 36 | 40 | 64 |
| Q heads | 32 | 40 | 64 |
| KV heads（GQA） | 8 | 8 | 8 |
| head dim | 128 | 128 | 128 |
| Q projection output | 4096 | 5120 | 8192 |
| K/V projection output | 1024 | 1024 | 1024 |
| Q:KV head ratio | 4:1 | 5:1 | 8:1 |

三者都使用 8 個原生 KV heads 與 head dim 128。因此 KV cache 保持 native-GQA，不展開為 Q-head 數量的 MHA cache；模型變大時，主要差異在 Q/O、FFN 的 GEMM 形狀與層數，而非 KV head 數量。

## 2. 共通設計：不因模型大小而改變的部分

三個 target 都維持相同的 QuaRot/PARD2 基本契約：

- W4A4KV4：權重 INT4、activation INT4、KV4；不改變 PARD2 的 proposal、驗證、acceptance 或 KV transaction 語意。
- KV cache：K 保持 rotation basis、V 保持原 basis；採 native-GQA append/decode。
- FFN：均採 `grouped_h256_v1`。三個 FFN width 恰好皆為 256 的倍數（12288=48×256、17408=68×256、25600=100×256），所以**不需要 padding**。
- exact small-chunk 數值契約：row-independent RMSNorm 用於避免不同 verifier row grouping 改變 reduction rounding；`QUAROT_FUSED_K1=0` 維持 AR/TI/TD 的已驗證數值行為。

## 3. HIP dispatch 差異

| 模型 | 主要線性層形狀（in → out） | HIP 路徑／調整 | 意義 |
|---|---|---|---|
| 8B | Q/O：4096；K/V：1024；FFN：12288 | N=4096 與 N=1024 的既有 LDS-swizzle 路徑；12288 走通用 B-prepack fallback | 基線實作，不需要 5120 或超大 FFN 專用分派。 |
| 14B | Q/O：5120；K/V：1024；FFN：17408 | 5120 使用既有 B-prepack path、1024 保持 LDS-swizzle；17408 走通用 B-prepack fallback | 已可正確且正式跑完，但沒有為 17408 新增專用 tile/wave 設定。 |
| 32B | Q：5120→8192；K/V：5120→1024；O/down：→5120；gate/up：→25600 | exact N=8192、N=5120、N=25600 B-prepack specialization；32B fused multi-projection 與 FFN 專用 waves | 這是唯一有專屬 shape dispatch 與 wave sweep 的模型。 |

### 32B 的實際調整

32B 的專用設定不是改變模型或 PARD2 架構，而是把已知靜態 GEMM 形狀直接映射到 HIP kernel：

```text
QUAROT_QWEN3_32B_GROUPED_NWAVES=4
QUAROT_QWEN3_32B_MULTI_NWAVES=2
QUAROT_FUSED_K1=0
```

- Q 投影的 output 是 8192（64×128），相對 8B/14B 的 4096/5120 顯著增大。
- FFN output 是 25600，使用 N=25600 專用 B-prepack dispatch；這也是 32B 每層最重的線性路徑。
- wave sweep 的結論為 grouped `g4` 優於 `g2`；multi `m2` 對核心 case 的合計改善約 0.42%。因此固定為 4/2，不再於 GPTQ 後重調 dimension。
- 32B 最終 HIP binary 的 correctness suite 為 178/178 passed。GPTQ 只改 INT4 codes，不改上述 shape、scale、A4/KV4 contract，因此沿用同一組 dispatch。

## 4. QuaRot rotation 與數值注意事項

| 模型 | hidden rotation | 已採取的數值處理 |
|---|---|---|
| 8B | 4096-wide Hadamard | 原 4096 路徑可維持既有 bit-exact TD 行為。 |
| 14B | generalized HadK：H40 × H128（width=5120） | TD 非 final tap 的 FP16 inverse HadK 中間值可能 overflow；採 inverse 前除以 16、inverse 後以 FP32 還原，final RMSNorm 亦用 FP32，並加 finite hard-fail。 |
| 32B | generalized HadK：H40 × H128（width=5120） | 同樣須以安全 restore 供 selected hidden-feature extraction；否則非 final tap 會出現 Inf。這是 collector/數值工具修正，不是 target 或 PARD2 架構改動。 |

這點是 14B/32B 相對 8B 最重要的**正確性**差異：5120 並非單一傳統 Hadamard width，且 feature restore 的 FP16 accumulation 範圍需要額外保護。

## 5. 對 PARD2 profile 的影響

| target | TI drafter | TD drafter | TD 四個 target taps 的拼接寬度 | 結論 |
|---|---|---|---:|---|
| 8B | PARD2-Qwen3-8B | PARD2-Qwen3-8B | 4×4096 = 16384 | 原始 canonical profile。 |
| 14B | PARD2-Qwen3-8B | PARD2-Qwen3-14B | 4×5120 = 20480 | canonical 14B profile；TD 使用 strict target alignment。 |
| 32B | PARD2-Qwen3-8B | 14B-on-32B 對齊實驗 | 4×5120 = 20480 | geometry 相容，但 14B warp 直接用於 32B 沒有 acceptance，故不是 canonical TD；需要對 32B 重新 calibration/training。 |

要點是：14B 與 32B 同為 5120 寬，**只代表 TD projection 的輸入 shape 同為 20480，不代表特徵分布或最佳 tap 完全相同**。32B 的層數（64 vs. 40）、Q:KV ratio（8:1 vs. 5:1）、FFN 寬度都不同，所以 14B TD warp 可作為初始化或診斷基準，但不可直接視為已對齊的 32B drafter。

## 6. 目前採用與未採用的調整

| 項目 | 8B | 14B | 32B |
|---|---|---|---|
| dedicated shape dispatch | 既有 4096/1024 | 無 17408 專用版本 | 有：5120/8192/25600 |
| dedicated wave overrides | 無 | 無 | 有：grouped=4、multi=2 |
| 5120 feature restore protection | 不需要 | 已採用 | 已採用（extractor/calibration） |
| 改 QuaRot/PARD2 演算法 | 否 | 否 | 否 |
| 仍值得獨立探索 | 以 acceptance/overhead 為主 | 17408 tile 是否有足夠收益 | 只在獨立 parity 測試下研究 K1 或其他 kernel micro-tuning |

## 7. 實務結論

1. **不要以參數量判斷 HIP 設定。** 真正決定 dispatch 的是每個 GEMM 的 `(M, N, K)`，例如 14B 與 32B 都是 hidden=5120，但 Q output、FFN width 和層數不同。
2. **14B 的重點是數值安全，32B 的重點是 exact-shape performance dispatch。** 14B 現有通用 kernel 已足以 qualification；32B 則因 8192/25600 的高頻大矩陣，有實測支持的專屬設定。
3. **保持演算法不變。** 目前的差異都限制在 profile validation、shape dispatch、wave selection 與 5120 feature restore；沒有為模型大小更換 QuaRot 或 PARD2 推論架構。

## 8. 依據與延伸閱讀

- 8B 整合與數值契約：[PARD2_INTEGRATION_REPORT_ZH_FINAL.md](PARD2_INTEGRATION_REPORT_ZH_FINAL.md)
- 14B target 的量化、TD numerical correction 與正式結果：[14B formal report](../qwen3_14b_results/formal_8k/REPORT.md)
- 32B kernel dimensions、waves、GPTQ 與 VRAM 結果：[QWEN3_32B_W4A4KV4_PARD2_REPORT_ZH.md](QWEN3_32B_W4A4KV4_PARD2_REPORT_ZH.md)
- 32B 對齊 14B TD drafter 的 S0–S2 診斷：[QWEN3_32B_PARD2_TD_S0_S2_RESULTS_ZH.md](QWEN3_32B_PARD2_TD_S0_S2_RESULTS_ZH.md)
