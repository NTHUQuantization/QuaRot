# 逐 row RMSNorm、activation INT4 packing 與 residual 實作說明

## 實測資料（primitive 範圍）

2026-09-24，在既有 ROCm 容器、Radeon AI PRO R9700 的協調 GPU 空檔執行。B=1、hidden=4096、FP16、clipping=0.9；每個 shape 交錯五種方法，40 組 samples、每組 10 次，以下是每次呼叫的 GPU-event median。事件涵蓋 eager dispatch 間隙，不等同純 kernel duration。

| M | norm+quant 未融合（ms） | norm+quant 融合（ms） | residual add + 已融合 norm（ms） | residual 全融合（ms） |
|---:|---:|---:|---:|---:|
| 1 | 0.075923 | 0.014296 | 0.018062 | 0.012830 |
| 15 | 0.069582 | 0.009582 | 0.013196 | 0.008664 |
| 16 | 0.070540 | 0.009584 | 0.013390 | 0.008678 |

三種 M 的實測 launch 數相同：norm+quant **7 → 1**；residual add + 已融合 norm **2 → 1**；完整未融合 residual+norm+quant 為 8 launches。全部比較的 packed bytes、scales 與 residual 均完全相等。

既有與新增測試合計 **85 passed（2.28s）**，見 `norm_tests.log`。primitive 原始資料、40 組 samples 與 extension SHA256 見 `norm_primitive.json`；15 份 `norm_M*_*.trace.json` 保留實際 launch 名稱。這組數據不能外推為整個 target verify 或 PARD-2 TPS 收益。

## Kernel 設計

新增 `rms_norm_quant_i4_rows_clipped(input, mean_dim, eps, clip_ratio)`。每個 row 使用一個 256-thread workgroup；M 僅改變 workgroup 數量，沒有跨 row reduction 或共用 row 的資料，因此保持 PARD-2 的 M=1 與 M=15/16 算術順序。normalized FP16 row 暫存在 LDS，無須先把完整 normalized tensor 寫回 global memory。

順序固定為：FP16 input → FP32 平方和 → 原本 `rms_norm_rows` 的 reduction tree → `rsqrtf` → **round 回 normalized FP16** → absolute max → **FP16 除 7** → checkpoint clipping ratio → **FP16 rounding、tiny clamp** → **FP16 除 scale** → round-to-nearest-even → clamp `[-8, 7]` → signed INT4 packing，第一個元素放在 low nibble。

不能把 FP32 normalized values 直接拿去量化，也不能把 `/7` 與 clipping 合併成一個 FP32 表達式；這些中間 rounding 會影響 INT4 threshold。新增路徑以正式 `quarot.nn.Quantizer` 為 oracle，驗證 packed bytes 與 scales 的完整相等，不使用誤差比例門檻。

舊的 `rms_norm_quant_i4_rows` 三參數 API 保留原來的 contract，供既有實驗重現；新的 runtime 路徑由 attention/MLP consumer 的 quantizer 讀取 checkpoint clipping ratio。`--fused-norm-quant` 保留為 opt-in，關閉時仍使用既有 row norm + quantizer。這個修改沒有變更 checkpoint、權重、原本非 exact-row 模式的 `FusedRMSNormQuant`，也沒有放寬驗證標準。

## Residual add 評估

另新增 `residual_rms_norm_quant_i4_rows(input, residual, mean_dim, eps, clip_ratio)`，回傳 `(packed, scales, residual_output)`。加法先 round 成 FP16，再做相同 RMSNorm；`residual_output` 保留後續 MLP skip 所需的 FP16 residual，兩個輸入不做 in-place 修改。

可考慮的 decoder 融合邊界是「attention output + 舊 residual → post-attention norm」。最終 MLP residual add 若跨到下一層 norm，會牽涉 decoder 輸出和 TD hidden-feature hooks；目前 primitive 的資料不足以支援這種跨層改動，故沒有接入 decoder。**Residual primitive A/B 不代表已取得 target verify 或 PARD-2 TPS 收益。**

## 重現與資料

`tests/test_verification_norm_quant.py` 涵蓋 M=1/15/16、hidden widths 128/4096/5120/8192、clipping 1.0/0.9/0.73、zero/subnormal/FP16 最大有限值、正負 half-integer 邊界、residual input 保持不變，以及 current-stream graph replay。舊 API 的 `tests/test_rms_norm_quant_i4.py` 仍須一併通過。

```bash
PYTHONPATH="$PWD:$PWD/third-party/hadacore" python -m pytest -q \
  tests/test_rms_norm_quant_i4.py tests/test_verification_norm_quant.py
PYTHONPATH="$PWD:$PWD/third-party/hadacore" python \
  benchmarks/verification_norm_quant_ab.py \
  --output verification_optimization_20260924/norm_primitive.json
```

primitive script 交錯各方法取樣，記錄 GPU event 樣本、median、kernel launches、完整 profiler traces 與 extension SHA256。正式 target verification latency 和 PARD-2 tokens/s 請以同目錄的 model A/B 報告為準；未生成或未通過的量測不能推估為收益。
