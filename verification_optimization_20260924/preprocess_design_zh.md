# H128 與 chunk 前處理設計

## Batched Q Hadamard

`verification_preprocess.hip::batched_h128_safe_rows_kernel` 新增專用 FP16 H128 kernel。每個 workgroup 仍最多處理 8 個有效 rows，對 WMMA fragment 其餘 rows 明確提供零，不直接啟用已知有 row 混用問題的 16-row backend。所有 workgroups 放在同一次 grid launch：M=16、32 Q heads 的 512 rows 使用 64 個 workgroups、一次 launch；fallback 仍以每 8 rows 呼叫原 Hadacore。

運算保留原路徑的 H16/4、FP16 中間結果，以及轉成 FP16 的 H8 正規化係數，再將最終結果存回 FP16。WMMA 每個中間元素均會寫入，無需 fill；有效 rows 在 load/store 處遮罩，無需額外 padding tensor；輸出直接寫入完整 tensor，無需 `cat`。這些是實作上的 launch / allocation 移除，速度收益另依 A/B 資料判定。

`QUAROT_BATCHED_H128=0` 可回到原安全分批路徑。非 H128、非 FP16 或 extension 尚未重建時也保留原路徑。

## Chunk Q / KV

這裡的 norm 是 **Qwen3 projection 後、逐 head 的 Q/K RMSNorm**；attention
之前的 hidden-state RMSNorm 仍在 Q/K/V projection 之前。流程是
`hidden RMSNorm → Q/K/V projection → Q/K head norm → RoPE → Hadamard`。
QuaRot 論文的 Llama attention 圖沒有這個 Qwen3 head norm，不能由該圖推定
本模型的 projection 後無 norm。模型端沿用 Transformers 的 Qwen3Attention，
本次未新增 norm 或修改 checkpoint。

原 K1 已融合 Q RoPE 與 K RoPE／Hadamard／KV 量化 append，但 Q/K head norm
在外部，Q Hadamard 也在後續處理；且原 K1 是 M=1、非 transaction 路徑。
新 chunk kernels 將這些前處理納入，支援 M=15／16 verification 與 provisional
KV 寫入。Grouped checkpoint 的原 M=1 K1 rounding 路徑仍優先保留。

新增兩個獨立 kernel 與 API：

- `chunk_q_norm_rope_hadamard`：逐 head Q RMSNorm → RoPE → 安全 H128 → FP16 Q。
- `chunk_k_norm_rope_append_i4`：逐 head K RMSNorm → RoPE → FP32 Hadamard → KV 非對稱 INT4 量化 → 指定 page/slot append，同一次 kernel 也量化及寫入 V。

Q 的 wave32 workgroup 處理最多 8 個攤平後的 `(batch, token, head)` rows，逐 row 完成 norm/RoPE 後沿用已驗證的 8-row H128 tile；KV 則以 128-thread workgroup 處理一個 head。兩者支援 QKV 合併 projection 產生的 strided views。K 的 page 與 slot 從 GPU metadata 的最終長度減去 chunk 長度，再加 token index 計算；因而同一 chunk 可跨 page，也可覆寫拒絕後的 provisional slots。

Qwen3 norm 讀取既有 runtime 的 FP16 weight，不改寫 checkpoint 或新增權重轉換。其語意依序為 FP32 平方與 mean、FP32 正規化、轉 FP16、FP16 weight 乘法。根據安裝的 PyTorch HIP `Reduce.cuh` 及原始 M=16 trace 的 `MeanOps<...,4,4>`，mean 使用每 lane 連續 4 個平方值依序相加，再使用 shuffle-down offsets 1、2、4、8、16 的 reduction 順序，避免一般 descending reduction tree 改變量化邊界。

嚴格中間值比對另外發現，這個 PyTorch/HIP 組合的 `rsqrt_wrapper` 呼叫 `::rsqrt(float)` 時，實際 overload 會先轉 FP64、完成 reciprocal square root 與 refinement，再轉回 FP32；直接替換成 `rsqrtf` 會讓部分 rows 的 inverse 相差 1 FP32 ULP，並在 HALF tie 或 INT4 邊界造成差異。以獨立 gfx1201 編譯器 IR 核對後，新 kernel 保留此 overload 行為，每 head 僅由一個 thread 計算並以 shared memory 廣播。同時在 FP32 mean、epsilon add、inverse 及正規化乘法後放置 register materialization boundary，維持 eager 多 kernel 之間原有的 FP32 rounding，避免融合後略過中間 rounding；這些 boundary 不產生額外 global memory traffic。

RoPE 使用 HIP `__hmul_rn` / `__hadd_rn` 保留兩次 FP16 product 及 FP16 add，明確禁止 compiler 將它們收縮成省略中間 rounding 的 half FMA。Q 使用安全 Hadacore 的 FP16 中間邊界與完整 wave32 workgroup；K 保留既有 `fused_append_kv_i4` 的 FP32 butterfly、FP16 scale/zero、FP32 除法及 ties-to-even rounding，沒有改成另一套量化公式。既有 KV fused append 的 scale clamp 位於除以 15 之後；新 kernel 亦保留該順序。全程使用 PyTorch current HIP stream，以支援 graph capture。

Runtime 的 `QUAROT_CHUNK_PREPROCESS=0` 保留原前處理。支援範圍為原生 GQA、FP16 H128、FP16 Q/K norm weights、無 attention mask、cache 已完成初始化；其他組態 fallback。

## 驗證

`tests/test_verification_preprocess.py` 包含 H128 任意 row 數與不同 rows impulse 的 bitwise 安全分批 oracle、M=1/15/16、strided QKV、B=2、跨 page、非順序 physical page index、非目標 layer/slot sentinel，以及零值/常數值 KV。H128、Q 輸出、KV packing 與 scale/zero 均要求 bitwise equality。完整模型 causal、拒絕回退與 PARD-2 tokens/s 由主 A/B harness 驗證。測試結果與速度數據應以實際執行紀錄為準。

修正 rounding 與 tile 後，前處理專項 41 個測試全數通過，見 `preprocess_rounding_tests.log`。曾失敗的初次開發紀錄亦保留，不能當作最終版本的驗證結果；最終不保留 debug API。`diagnostics/` 內的 rsqrt overload fixture、LLVM IR 與重現命令可獨立核對 FP64 promotion 的原因。
