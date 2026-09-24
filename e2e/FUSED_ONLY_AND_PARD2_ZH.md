# Fused-only 與 PARD-2：執行方式及路徑

目前支援關閉 PARD-2，只使用 fused W4A4KV4 target：指定 `--mode ar`。
AR 不載入 draft model、不產生候選 token，也不執行推測驗證與拒絕回退。
它保留 fused target 的 INT4 GEMM、KV cache 與適用的融合 kernels。

|模式|Target|Draft model|生成方式|
|---|---|---|---|
|`ar`|相同 fused W4A4KV4 target|不載入|Prefill 後逐 token 解碼|
|`pard2-ti`|相同 fused W4A4KV4 target|載入|TI 提案、target 驗證、接受／回退|
|`pard2-td`|相同 fused W4A4KV4 target|載入|TD target features、提案、驗證、接受／回退|

目前生成介面為 batch size 1、greedy。以下使用已量測的 Qwen3-8B legacy
checkpoint；14B／32B 必須另外指定相符路徑與 profile，見
[Qwen3 完整指南](PARD2_QWEN3.md)。容器名稱含 `32b` 不代表下列指令載入 32B。

## 主機、容器與資料路徑

Git repo 是下列 `fused_v1` 目錄，分支 `fused_pard2`；上層 `HIP_Fusion`
是工作區。表中的絕對路徑是本次機器配置，其他機器須調整 mount／CLI 路徑。
模型、tokenizer、資料集及編譯後的 `.so` 不包含在 Git 內。

|用途|主機路徑|`qwen3_32b_quarot_clean` 容器內路徑|
|---|---|---|
|工作區|`/user/undergraduate/wfching25/HIP_Fusion`|`/workspace_root`|
|Git repo／執行目錄|`/user/undergraduate/wfching25/HIP_Fusion/fused_v1`|`/workspace_root/fused_v1`|
|Qwen3-8B target|`/user/undergraduate/wfching25/HIP_Fusion/qwen3_8b_fused_v1_rtn_w4a4kv4`|`/workspace_root/qwen3_8b_fused_v1_rtn_w4a4kv4`|
|Hugging Face cache|`/user/undergraduate/wfching25/HIP_Fusion/.hf_cache/pard/hub`|`/workspace_root/.hf_cache/pard/hub`|
|Benchmark 資料根目錄|`/user/undergraduate/wfching25/HIP_Fusion/QuaRot/.benchmark_data/amd_pard_6f279bf`|`/workspace_root/QuaRot/.benchmark_data/amd_pard_6f279bf`|

在上述 HF cache 根目錄下：

- Tokenizer：`models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218`。
- TI／TD drafter：`models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135`；AR 不需要載入此模型。

## 直接生成：關閉或開啟 PARD-2

從主機進入既有環境；下列命令需要已建置的 HIP extension，建置方法見
[reproduction guide](../verification_optimization_20260924/README.md)。

```bash
docker exec -it qwen3_32b_quarot_clean bash
cd /workspace_root/fused_v1

# 共用目前最佳化開關，只改 mode；這支 launcher 也支援 AR。
bash verification_optimization_20260924/run_optimized_pard2.sh \
  --mode ar --prompt 'Explain speculative decoding briefly.' \
  --max-new-tokens 256 --json-output /tmp/fused_ar_generation.json

bash verification_optimization_20260924/run_optimized_pard2.sh \
  --mode pard2-ti --prompt 'Explain speculative decoding briefly.' \
  --max-new-tokens 256 --json-output /tmp/fused_ti_generation.json

bash verification_optimization_20260924/run_optimized_pard2.sh \
  --mode pard2-td --prompt 'Explain speculative decoding briefly.' \
  --max-new-tokens 256 --json-output /tmp/fused_td_generation.json
```

以上使用 `e2e/pard2.py` 的本機 8B 預設路徑。換機器時請明確傳入
`--target`、`--tokenizer`，以及 PARD-2 所需的 `--draft`。正常生成保留 EOS；
不要將 benchmark 的 `--ignore-eos` 誤認成生成必需選項。
`/tmp/...` 是容器內輸出路徑；需由主機直接讀取時，改寫到
`/workspace_root/fused_v1/` 下的自訂結果目錄。

Launcher 執行 `python -m e2e.pard2 --compile-mode eager --fused-norm-quant`，
並將下列四個環境開關預設為 `1`。`--compile-mode eager` 控制 drafter
的編譯方式，並不等於關閉獨立的 target verification graph。

|開關|功能|Fallback|
|---|---|---|
|`QUAROT_BATCHED_H128`|安全 8-row workgroup 的單次 batched H128 launch|設為 `0`|
|`--fused-norm-quant`|逐 row RMSNorm／clipping／scale／INT4 packing|`--no-fused-norm-quant`|
|`QUAROT_STATIC_KV_METADATA`|固定且跨層共用的 GPU KV metadata|設為 `0`|
|`QUAROT_VERIFICATION_GRAPH`|符合條件的 M=15／16 target forward graph|設為 `0`；須搭配 static metadata 才會使用|
|`QUAROT_CHUNK_PREPROCESS`|符合條件的 Q／KV norm、RoPE、Hadamard、KV append|設為 `0`|

AR 的 M=1 decode 不使用 M=15／16 verification graph；其餘融合依 shape／
checkpoint 條件分流，關閉 PARD-2 不等於關閉 fusion。原有 grouped checkpoint
M=1 K1 路徑仍保留。各開關是可回退選項，不會更換權重或放寬驗證標準。

## Benchmark 與量測狀態

在容器的 repo 目錄下，以下是單模式 fused AR 診斷量測；換成
`--mode pard2-ti`／`pard2-td` 並指定不同輸出檔即可比較同一配置。
GPU 閒置時才執行，模式須依序跑。

```bash
export PYTHONPATH="$PWD:$PWD/third-party/hadacore"
export QUAROT_BATCHED_H128=1
export QUAROT_STATIC_KV_METADATA=1
export QUAROT_VERIFICATION_GRAPH=1
export QUAROT_CHUNK_PREPROCESS=1
python -m e2e.benchmark_pard2 \
  --mode ar --dataset humaneval --limit 3 --generated-tokens 256 \
  --warmups 2 --sweeps 3 --compile-mode eager --ignore-eos \
  --fused-norm-quant --output /tmp/fused_ar_benchmark.json
```

上述是小樣本效能診斷，非正式三資料集 qualification。需要分開記錄 prefill、
decode、E2E，並自動等待閒置 GPU，請使用
[AR／TI／TD／AR-repeat 序列量測腳本](../pard2_speedup_20260924/README.md)。
截至本次整理，該比較因 GPU 忙碌等待一小時後逾時，**沒有新的 speedup 結果**。
既有 verification report 的加速比是「PARD-2 最佳化前後」，不是「有／無 PARD-2」。

## 程式與報告入口（相對 repo 根目錄）

|用途|路徑|
|---|---|
|生成 CLI／路徑預設|[`e2e/pard2.py`](pard2.py)|
|AR／TI／TD runtime、draft 載入分支|[`e2e/speculative.py`](speculative.py)|
|正式 benchmark 與 provenance|[`e2e/benchmark_pard2.py`](benchmark_pard2.py)|
|Target graph capture／replay|[`e2e/verification_graph.py`](verification_graph.py)|
|KV cache、metadata、transaction|[`quarot/transformers/kv_cache.py`](../quarot/transformers/kv_cache.py)|
|已完成的 verification A/B 數據、kernel 設計與引用|[`verification_optimization_20260924/report_zh.md`](../verification_optimization_20260924/report_zh.md)|
|建置、正確性測試、A/B 重現|[`verification_optimization_20260924/README.md`](../verification_optimization_20260924/README.md)|
|目前 AR 對 PARD-2 phase benchmark|[`pard2_speedup_20260924/README.md`](../pard2_speedup_20260924/README.md)|

主機上的完整文件路徑例如：
`/user/undergraduate/wfching25/HIP_Fusion/fused_v1/e2e/FUSED_ONLY_AND_PARD2_ZH.md`；
容器內對應 `/workspace_root/fused_v1/e2e/FUSED_ONLY_AND_PARD2_ZH.md`。

## 本次整理檢查

76 個 CPU 分流／speculative／benchmark 合約測試通過（以
`HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1` 執行）；Python AST、JSON、
launcher 的 `bash -n` 與 `git diff --check` 通過。報告由保存的量測 JSON
重新產生。既有 GPU correctness 紀錄為 229 passed，見
[`final_tests.log`](../verification_optimization_20260924/final_tests.log)；
本次文件整理沒有重新量測 GPU 效能。
