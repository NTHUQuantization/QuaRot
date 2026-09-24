# KV metadata 與 verification graph 設計

`QUAROT_STATIC_KV_METADATA=1` 在 cache 建立時為 M=1、15、16 配置
position、append 與 causal metadata。每種 M 的 tensor 位址固定，全部 decoder
layers 共用同一組。每次 forward 以 GPU base-length scalar 啟動一次 metadata
kernel，產生 positions、每個 query row 的有效 page 數與最後一頁 offset。
`kv_indices` 配置最大容量，但 `indptr` 只指向目前有效的部分，未提交或過期
KV slots 不會被 attention 讀取。append 使用真實 batch metadata，attention
使用 B×M 虛擬 batch，令第 t 個 query 只看 prefix+t+1 個位置。

回退只修改既有的 logical cache length；下一次 forward 重新更新 GPU metadata。
跨 page、全拒絕及部分接受沿用相同機制，不搬移 KV，不改 acceptance 規則。
其他 M、有 attention mask 或停用開關時保留原 metadata 路徑。

`QUAROT_VERIFICATION_GRAPH=1` 配合 static metadata，為首輪 M=15 與一般 M=16
各建立一張 graph。固定 input IDs、GPU base length、positions、KV pages 與
metadata 位址；graph private pool 保留中間 tensor 與 output 位址。replay 前
只更新 IDs 與 base length。生成迴圈在此路徑省去重複的 target `arange`。
graph 不捕捉 draft、acceptance 或 commit 決策。graph output 是可覆寫的
持久 buffer，驗證腳本在 replay 前複製所需比較資料。

TD 的 Python forward hooks 只在 capture 執行，因此另保留每張 graph 的 hidden
tap tensor references，replay 後恢復 collector 對應；feature 還原、projection
與接受 rows 的選擇維持原流程。不同 generate 呼叫共用 target cache 與 graph，
重新 prefill 前重設 logical length 與各層初始化狀態。

GEMM、INT4 quant/dequant、FlashInfer append/decode 的舊 launch 使用 default
stream；本次改成 `at::hip::getCurrentHIPStream()`，讓它們與 PyTorch 的 graph
capture stream 一致。這是 graph 正確性修正，沒有更換 GEMM tile、運算量或權重。

graph 首次建立包含兩次 warmup 與 capture，`capture_ms` 另行記錄。
正式量測欄位須註明是否已暖機，不能把未攤銷的初始化成本隱藏成首次呼叫收益。
ROCtracer 7.2 在此環境會漏記 graph child kernel events，因此 graph 的 GPU
kernel 數改以 `hipGraphGetNodes`／`hipGraphNodeGetType` 遞迴列舉，再加上
graph 外的直接 kernel launch；不完整 trace 數另存，不能當作實際 launch
總數。CPU 的 `hipGraphLaunch` 次數另列，單次 replay 不是單一 GPU kernel。

測試入口：`tests/test_verification_metadata.py`，以及 `target_probe.py` 的
M=1/15/16、contexts=127/128/129、causal future-token 與 rollback 比較。
