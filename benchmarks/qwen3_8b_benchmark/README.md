# Qwen3-8B GPTQ 通用測速套件

指定自己的 checkout 與版本名稱，即可獨立執行相同 workload 的 prefill、decode、E2E 測速。本套件沒有指定實驗分組、分支順序或必須先測的基準版。`compare.py` 可比較任意兩個或多個完整結果。

**可比較的前提**是相同模型權重、input IDs、生成與計時配置，以及可核對的硬體／軟體環境。工具會檢查這些條件與輸出 token；相同配置不保證不同實作的數值輸出完全相同。不同輸出的結果仍可比較自然生成耗時，但不能宣稱等輸出純加速。

## 快速開始

```bash
git clone --branch benchmark/qwen3-8b-ablation-gptq-handoff --single-branch \
  https://github.com/NTHUQuantization/QuaRot.git
cd QuaRot/benchmarks/qwen3_8b_benchmark
cp config.example.json config.local.json
```

修改 `config.local.json`：

|欄位|用途|
|---|---|
|`label`|自訂版本名稱，例如 `my-kernel-v2`；比較的名稱需各自不同|
|`run_id`|本次執行 ID，重跑請用新 ID|
|`repo`|要測的 checkout 路徑；不限制 Git branch 或 commit|
|`engine`|內建 `ar`、`pard2-ti`，或舊 cache API 的 `legacy-ar`|
|`target`|共用 Qwen3-8B GPTQ checkpoint|
|`tokenizer`|固定版本的 tokenizer 目錄|
|`draft`|PARD2-TI 所需的固定 BF16 draft；一般 AR 可省略|
|`fht_path`|已建置的 `fast_hadamard_transform` 所在目錄|
|`output`|結果根目錄，最後寫入 `<output>/<run_id>/<pilot或formal>/`|
|`adapter`|選填：其他 runtime 的 Python adapter 檔案，介面見下文|

相對路徑以 config 所在目錄為準。Docker 使用容器內路徑。修改版本只需換 `repo`、`label`、`run_id`，不要修改 profile、prompts 或計時 worker。

```bash
# 修改 config 後，在相容 ROCm 開發環境建置指定 checkout。
python prepare.py --config config.local.json

# 僅 CPU 檢查 checkpoint、完整 input hashes、Git 與 extension 檔案。
python run.py --config config.local.json --check

# 先測小樣本並檢視報告，再執行完整測量。
python run.py --config config.local.json --stage pilot
python run.py --config config.local.json --stage formal
```

`runtime/` 附一份可選用的 runtime 原始碼快照，範例 config 預設使用它；也可自行指定任何相容 checkout。`reference/runtime_sources.json` 保存快照檔案 SHA-256，實際所測 checkout 的 source、extension 與 adapter hashes 都會記錄。沒有提供預編譯 `.so` 或模型權重。

每次命令只執行一個版本；可用 tmux 保持終端。不要同時在同張 GPU 測多個版本。腳本拒絕覆寫或接續半份結果；失敗目錄保留作排查，使用新 `run_id` 重跑。建立結果目錄內的 `STOP` 檔可在下一個 request 前停止。測量時不要修改 runtime、adapter、權重或 profile。

## 相容環境與版本

參考環境：Linux、Python 3.10、AMD gfx1201 32 GiB、ROCm/HIP 7.2、PyTorch `2.9.1+git5bc97ba`、Transformers `4.57.6`、ROCm FlashAttention `2.8.3`、safetensors `0.7.0`。需有 `rocm-smi`，GPU 測量前應無其他工作。

`reference/environment.json` 保存 pip freeze 與容器 image ID，包含本機 wheel 路徑，**不是通用 pip requirements 或可直接 pull 的 image tag**。組員需使用相同 ROCm builds／既有環境映像；不能以同版本 CUDA wheel 取代。`prepare.py` 為 gfx1201 建置，其他架構需確認 kernel 支援並自行建置，報告為不同硬體的結果。

入口固定 `OMP_NUM_THREADS=4`、`QUAROT_BATCHED_H128=1`、`QUAROT_STATIC_KV_METADATA=1`、`QUAROT_VERIFICATION_GRAPH=1`、`QUAROT_CHUNK_PREPROCESS=1`。請取消其他 `QUAROT_*` tuning 環境變數，不要使用 `python -O`。

內建 adapter 的適用範圍：

- `ar`／`pard2-ti`：checkout 需提供 `e2e.speculative.load_runtime`、`FusedPardRuntime`、phase adapter 與 runtime provenance 介面；隨附快照符合此介面。
- `legacy-ar`：針對舊 Qwen3 cache API 的 QuaRot checkout，在記憶體中補上 Transformers 4.57 cache 呼叫／checkpoint 欄位相容性，保留原 kernel 與 rounding。套用的 diff 會保存，若 source pattern 不符合則拒絕執行。
- 任意其他 runtime：提供 `adapter`，遵守下述協定；不代表所有任意倉庫都能直接套用內建 adapter。

## 固定模型與輸入

本 profile 只處理 **Qwen3-8B GPTQ W4A4KV4**，不是任意模型尺寸的通用 profile。

- Target：共用 GPTQ checkpoint，grouped_h256_v1，包含 config、index、36 layer shards 與 shared shard。
- Tokenizer：`Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`。目錄需包含 hash 清單列出的 tokenizer/config/index 檔；AR 不需要 BF16 base 權重。
- Draft（使用時）：`amd/PARD2-Qwen3-8B` revision `67a1516c8f6fc145cda99916799a0cbb3a4af135`，含 `warp_model.bin`、`config.json`、`model.safetensors`。

模型不隨 Git 分享，請取得共同權重檔案。`reference/input_hashes.json` 記錄檔名→SHA-256；入口在測量前後核對 hash，並檢查 GPTQ shard signature、index 完整性與檔案長度。RTN 或另一份 GPTQ 權重不能混入這個 profile。

`reference/prompts.jsonl` 包含固定公開資料集 prompts、input IDs、來源檔案 hashes／題號。HumanEval、GSM8K、MATH500 各 formal32題、pilot4題，formal共96題、pilot共12題，互不重疊；不要重新抽題、改 chat template 或重新 tokenization。抽樣方式與三次 sweep 的確切順序在 `reference/profile.json`。

## 計時與記憶體配置

|項目|固定值|
|---|---|
|生成|batch1、greedy、thinking=false、自然 EOS；輸出數包含生成的 EOS|
|上限|pilot新增128 tokens；formal新增256 tokens|
|精度|target FP16；KV INT4；使用 draft 時為 BF16|
|量化配置|activation clip0.9、KV clip1.0／group128、grouped_h256_v1|
|Cache|page128、capacity8192；重用儲存空間，每 request 重置邏輯狀態|
|PARD2-TI（使用時）|draft_k15、adaptive_k=false、AR fallback0、eager draft、verification graph 開啟|
|重複|3 sweeps，每題各量 prefill、decode、E2E|
|Warmup|所有 prefill shapes；每 phase 三個代表題（每資料集一題）；graph 預熱，新 capture 視為無效|

**Prefill** 從 request cache／prefix 準備開始到 prefix 完成，停在第一次 token selection 前。若 runtime 需要 draft，其必要的 prefix 準備也要包含。

**Decode** 先在計時外建立乾淨 prefix，再從第一次 token selection 計到完成 committed output IDs。

**E2E** 是獨立測量整段 prefill+decode，中間沒有額外同步，不以兩次獨立 phase 測量相加取代。

使用 `time.perf_counter()` wall time，外部計時起迄以 `torch.cuda.synchronize()` 同步；包含 host 控制、launch、token selection。排除模型載入、tokenization、input GPU 搬移、編譯、warmup、驗證、logging、統計。此 worker 的同步／計時／memory reset 次序沿用既有方法。

每 phase 初始同步後 reset peak memory；decode 的 prefix 已在 reset 外完成。記錄 baseline allocated/reserved、peak allocated/reserved、incremental peak allocated，含常駐模型與 buffer，**不是整卡 VRAM**。每 request 保存 GPU 使用率、溫度、時脈、power／PID 快照；未鎖定 GPU clocks，連續版本可能有溫度／時脈漂移。

## 結果與任意版本比較

```text
runs/<run_id>/formal/
  contract.json / manifest.json       # 配置識別碼、實際 paths、commit、source/extension/adapter hashes、環境
  prompts.jsonl / profile.json
  runs.jsonl                          # 每個 phase 的原始 timing、memory、output IDs
  gpu.jsonl / qualification.jsonl
  warmup.json / generated_adapter.py
  complete.json / validation.json
  per_prompt.csv / summary_by_dataset.csv
  report_zh.md
```

pilot 每版108列、formal每版864列。每個版本同版跨 repeats／decode／E2E 必須逐 token 一致；不合格則不能進入有效比較。時間與 tokens/s 先取每題3次 median，再取跨題 median；**p10、p90、p99** 以每題 median 分布做線性插值，另外明列 pooled repeats percentiles。p99 只有少量題目，只作描述，不視為穩定服務尾延遲。

```bash
python compare.py runs/my-version-001/formal runs/another-version-001/formal \
  --output comparisons/formal-001
```

可在命令中列出任意多個結果資料夾。工具重新驗證 raw records，要求同 stage、同 `comparison_id`（profile、prompts、target/tokenizer、計時 worker、固定環境配置），並核對 GPU 型號、CPU 型號、Python、PyTorch/HIP 與依賴版本。不同 checkout、commit、名稱、runtime engine、adapter 是被比較的實作資訊，不要求相同，但會保存。

跨環境預設拒絕比較；若刻意比較硬體／軟體環境，使用 `--allow-environment-difference`，報告會明確標記，不能把差異全歸因於程式。即使型號相同也不保證同時脈／功耗／系統負載，需檢視快照並保持測量環境一致。

報告逐題檢查 output IDs／長度並保存首次差異位置。輸出一致才標示 equal-output speedup；不一致則標示 natural-generation time ratio，保留全部題目。E2E ratio 是左版時間÷右版時間，逐題配對幾何平均，overall三資料集等權；95% CI 使用2,000次 paired bootstrap、seed0。這是速度測試，不是答對率評估，token cap 可能截斷答案。

## 自訂 adapter 協定

在 config 指定 `"adapter": "/path/to/adapter.py"`。模組需提供：

```python
def load(config, artifact_dir):
    # 載入指定 checkpoint，遵守 profile；不得在這裡改 prompts 或測量 worker。
    # 回傳 runtime，至少包含 repo (Path)、eos_ids、_verification_graphs (dict)。
    ...

def phases(runtime, prompt, tokens):
    # prompt 是 GPU LongTensor [1, input_length]；tokens 是本 stage 上限。
    # 回傳物件，具 begin() / finish(state) / generated_source。
    ...

def provenance(runtime):
    # 回傳可 JSON 序列化的實際 dtype、cache、flags、source/extension hashes。
    ...
```

`begin()` 重設 request state 並完成 prefix，停在第一次 token selection **前**，回傳可 `close()` 的 state；`finish(state)` 完成自然生成，回傳 dataclass 或 SimpleNamespace，至少有 `output_ids: list[int]`。PARD 類實作應另提供 proposals／accepts／forward 次數等 raw stats。`generated_source` 保存實際 phase 邏輯。模型載入只執行一次，cache 儲存空間可重用，每個 request 的邏輯 state 必須重設。

worker 管理同步、計時、memory reset、warmup 與輸出驗證，adapter 不得插入另一套 timer 或偷偷改 workload。自訂 runtime 需審查 phase 邊界、精度與設定符合 profile；hash 一致只保證資料與 worker 相同，無法自動證明 adapter 完全遵守計算契約。`manifest.execution` 必須列出實際設定與額外來源，供結果使用者核對。

## 驗證

```bash
python -m unittest discover -s . -p 'test_benchmark.py' -v
python summarize.py runs/my-version-001/formal
```

套件測試使用 synthetic CPU fixtures，檢查 raw result 稽核、分位數、不同名稱比較、配置／環境不一致拒絕、輸出差異註記，以及以模擬 GPU API 執行完整 worker 流程；不把模擬結果當作測速資料。這份通用包尚未在新 checkout 重新執行 GPU 端完整測量。
