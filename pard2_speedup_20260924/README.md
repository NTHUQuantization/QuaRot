# Current fused AR versus PARD-2：prefill／decode／E2E

**狀態：尚未完成量測。** 2026-09-24 的首次執行在載入 AR 前等待 GPU 閒置，
一小時後逾時退出；沒有 AR／TI／TD 數據，也沒有可宣稱的 speedup。
腳本不會在逾時後自行重新啟動。既有 verification 最佳化報告比較的是
PARD-2 前後版本，不能當作這項比較的結果。

## 路徑與重現

- 主機 repo：`/user/undergraduate/wfching25/HIP_Fusion/fused_v1`。
- 容器：`qwen3_32b_quarot_clean`；repo mount：`/workspace_root/fused_v1`。
- 模型／tokenizer／資料集：沿用 8B 預設值，完整路徑見
  [執行與路徑指南](../e2e/FUSED_ONLY_AND_PARD2_ZH.md)。這支 runner 尚無 14B／32B 路徑選項。
- `run.py`：依序跑 AR、TI、TD、AR repeat，使用相同最佳化配置。
- `benchmark_phases.py`：在原 benchmark 的 HIP events 上增加 prefill span 記錄。

```bash
docker exec qwen3_32b_quarot_clean bash -lc '
  set -e
  cd /workspace_root/fused_v1
  python pard2_speedup_20260924/run.py \
    --output pard2_speedup_20260924/results_retry \
    --limit 3 --generated-tokens 256 --warmups 2 --sweeps 3 \
    --idle-timeout 3600
'
```

每次使用新的輸出目錄；runner 拒絕覆蓋已存在的模式結果。
上述結果可在主機
`/user/undergraduate/wfching25/HIP_Fusion/fused_v1/pard2_speedup_20260924/results_retry/`
讀取。目錄內包含 `ar.json`、`pard2-ti.json`、`pard2-td.json`、`ar_repeat.json`、
每模式的 log／GPU 狀態紀錄及 `manifest.json`。載入模型前才寫 manifest；
若在首次等待便逾時，可能只留下 `ar.idle_preflight.json`。

量測前須連續三次無 GPU process 且 utilization=0；等待不載入模型。
量測時每五秒抽查 GPU process，多於一個時只終止自身 benchmark，視為失敗。
這不是 GPU 排他鎖，抽查間仍可能漏掉短暫競爭；解讀結果須連同監控紀錄檢查。

## 固定 protocol 與欄位定義

Qwen3-8B legacy W4A4KV4、HumanEval 前 3 prompts、每次 256 generated tokens、
ignore EOS、2 warmups、3 sweeps、eager drafter。四個 `QUAROT_*` 最佳化開關
皆為 1，所有模式皆有 `--fused-norm-quant`。這是小樣本診斷，非正式 qualification。

|指標|定義與限制|
|---|---|
|`prefill_gpu_span_ms`|第一個 target prefill event 開始到最後一個 prefill event 結束；PARD 包含 draft prefix prefill 及中間工作，TD 包含 target feature materialization|
|`prefill_input_tokens_per_s`|原始 prompt token 數／上述 span；PARD 的分母包含額外 draft 工作，不是純 target prefill|
|`stage_ms.phase_target_prefill_gpu`|只計第一個 target prefill event|
|`stage_ms.phase_draft_prefill_gpu`|Draft prefix prefill events 的加總；AR 為 0|
|Steady decode tokens/s|沿用原 benchmark 的 wall timer，排除整個首輪 emitted tokens 及時間|
|E2E tokens/s|Generated tokens／generate wall time；包含 prefill、首輪及後續 decode，排除載入、tokenization、timer 前的 cache 建構及暖機時 graph capture|

PARD 首輪還需提案與驗證，所以 TTFT 不能當成純 prefill。Prefill 是 HIP
event span，其餘是 wall time，且 steady 排除首輪，不能直接將兩者相加為 E2E。
Wrapper 使用既有 events／同步，只增加量測後的 CPU scalar 處理；這些 CPU
處理位於原 generate timer 範圍內，會有少量額外負擔。

完成後須以 `(sweep, prompt_index, input_ids_sha256)` 配對，逐 token 檢查
AR／TI／TD／AR-repeat 的 256-token 輸出，再報告 paired medians 與 AR repeat 漂移。
Prefill speedup 定義為 `AR ms / PARD ms`；decode／E2E throughput speedup 為
`PARD tokens/s / AR tokens/s`。目前腳本只收集資料，尚未產生這些比較表。
