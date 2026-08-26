# fused_v1 組員更新整合：衝突處理與決策紀錄

> 分支：`fused_v1_PARD`
> ours：`34ff444`（PARD2 整合與最佳化）
> theirs：`origin/fused_v1@cc4ffca`
> merge commit：`c5a7907`
> 共同祖先：`923852e`

## 1. 目的與決策原則

這次整合的目標不是單純讓 Git conflict 消失，而是同時保留：

1. PARD2 已通過 qualification 的 speculative transaction、small-chunk verifier、TD hidden taps 與 AR/TI/TD exact parity。
2. 組員分支帶來的 native GQA、K1 fused decode、grouped-scale GEMM、interleaved projections、fused RMSNorm、streaming converter 與 Llama 3.1 支援。

相容性衝突依以下順序判斷：

1. 潛在效益較高者優先。
2. 效益相近、證據不足或會破壞 correctness 時，保留 ours。
3. 高效路徑若尚未滿足 parity，保留為明確可控的 opt-in，不直接刪除。
4. checkpoint 語意不可由檔案缺少 metadata 而被靜默改變。

## 2. Incoming 更新概況

組員分支包含三個主要 commits：

| Commit | 主要內容 |
|---|---|
| `65cd3fb` | decode correctness 修正、per-layer quantization |
| `1fe5859` | native GQA、Llama 3.1 8B 支援 |
| `cc4ffca` | K1 fused kernel、fused RMSNorm、interleaved projections、其他 runtime 最佳化 |

高潛力項目均先保留並進入測試，而不是因與 PARD2 分支重疊就直接選 ours。

## 3. 衝突一：舊 PARD checkpoint 與新 grouped-H256 checkpoint

### 問題

PARD2 目前使用的 Qwen3 target 是舊 checkpoint：

- FFN 使用完整 HadK basis。
- activation clip ratio 為 `1.0`。
- down projection 是 `OnlineHadamard → Quantizer → Linear4bit`。
- layer RMSNorm 使用 plain RMSNorm。

組員的新 converter/runtime 改為：

- grouped-H256 FFN。
- activation clip ratio `0.9`。
- padded/direct down projection。
- layer fused RMSNorm + quantization。

若直接採用 incoming runtime，舊 checkpoint 會在沒有 weight shape error 的情況下被套上不同量化語意，屬於 silent corruption。

### 解法：雙格式契約

runtime 依 checkpoint metadata 明確分流：

| 格式 | Metadata | FFN | Clip | Layer norm |
|---|---|---|---:|---|
| legacy | 無 version，或明示 `legacy_hadk` | exact HadK sequential down | 1.0 | plain RMSNorm |
| v2 | version `2` + `grouped_h256_v1` | grouped-H256 direct down | 0.9 | fused RMSNorm 可用 |

不支援的 version、format 或 clip 組合會 fail-fast。新 streaming converter 固定輸出 v2 metadata；既有 PARD target 不需 migration。

### 驗證

- Pinned Qwen3 checkpoint 判定為 `legacy_hadk`。
- 614 個 checkpoint keys 與 614 個 model keys 完全匹配。
- 無 missing、extra 或 shape mismatch。

## 4. 衝突二：PARD cache transaction 與 incoming persistent metadata/K1/graph

### Ours 必須保留的語意

- `begin/commit/rollback` speculative transaction。
- q_len 1–16 provisional chunk append。
- virtual causal verification metadata。
- commit 只保留 accepted prefix 與 correction token。
- expanded-MHA correctness oracle。

### Incoming 值得保留的能力

- persistent equal-length KV metadata。
- decode K1 fused RoPE + native-GQA KV4 append。
- CUDA/HIP graph 固定 metadata。

### Union 設計

- transaction 改變 logical cache length，persistent metadata 每次依目前 length 選 row，因此 rollback 不需重建 metadata。
- graph metadata 與 transaction 明確互斥。
- K1 在 transaction、attention mask、expanded-MHA、初始 cache layer 或 q_len > 1 時自動停用。
- capacity 在更新 logical length 前檢查。
- native GQA 永久保存原生 KV heads；expanded-MHA 仍可透過開關作 oracle。

## 5. 衝突三：K1 fused decode 的速度與 PARD parity

### 實際問題

Incoming 將 K1 預設開啟後，AR 使用 fused RoPE/KV append，但 PARD verifier transaction 為維持 provisional cache correctness 會停用 K1。真實 MATH prompt 14 測試中：

- AR 與 TI 都生成 256 tokens。
- 第一個差異出現在 generated token 14：AR `264`、TI `279`。

這表示 kernel 單獨的近似 correctness 測試不足以保證 speculative 與 AR 的 exact greedy parity。

### Ablation

設定 `QUAROT_FUSED_K1=0` 後，AR 前 32 tokens 與 TI 完全一致；因此問題定位到 K1，而不是 fused projections。

### 最終決策

- legacy checkpoint：K1 預設關閉，保護已 qualification 的 PARD 主線。
- grouped-v2 checkpoint：K1 預設開啟，保留新格式的效益。
- `QUAROT_FUSED_K1=0/1` 可明確覆寫。
- low-level cache API 仍保留 K1，後續可在 kernel 達到 row/chunk exact 後重新評估 legacy default。

這項決策沒有接受「較快但破壞 parity」的結果；高效路徑仍保留，不阻塞後續修正。

## 6. 衝突四：grouped-scale GEMM 改變舊 target 的數值路徑

Incoming `Linear4bit.forward` 直接輸出 FP16：

```text
INT4 GEMM + FP32 scale epilogue + final FP16 cast
```

舊路徑則是：

```text
INT4 GEMM → INT32 tensor → FP16 intermediate → dequantization
```

兩者單層 top-1 相同且誤差很小，但完整 autoregressive model 會累積 rounding 差異。MATH prompt 14 相對舊 formal AR 的第一個 token 差異出現在 generated token 20。

### 效益與決策

相同 32-token focused ablation：

| 路徑 | Steady tok/s | 相對舊 GEMM |
|---|---:|---:|
| 舊 INT32 + dequant | 19.713 | 1.000× |
| grouped-scale，關閉 fused projections | 24.682 | 1.252× |
| grouped-scale + fused projections | 25.480 | 1.293× |

因 grouped-scale 帶來約 25% 的直接收益，而且 AR/TI/TD 共用此路徑後仍 exact parity，故依「潛在效益優先」保留為預設。

同時新增 `QUAROT_GROUPED_SCALE_GEMM=0` fallback。搭配 `QUAROT_FUSED_PROJECTIONS=0` 時，真實 target 前 32 tokens 可逐 token 重現舊 formal AR，讓舊數值路徑仍可作 oracle 與回歸分析。

## 7. 衝突五：fused projections 與 checkpoint key 相容性

Q/K/V 與 gate/up 只在 execution time 共用 packed activation 與 multi-output kernel；checkpoint 仍維持個別 projection keys，不做 weight migration。

- 預設由 `QUAROT_FUSED_PROJECTIONS=1` 開啟。
- `QUAROT_FUSED_PROJECTIONS=0` 回到各 projection 獨立執行。
- interleaved storage 僅重新安排 runtime views，2/3 projections、M=1/15/16 均與 separate projections bit-exact。

## 8. 衝突六：converter resume 與 RNG

Incoming streaming RtN/GPTQ 支援 shard resume，但 rotation matrix 原先依賴 ambient RNG。若程序中斷後 caller RNG state 不同，resume 可能產生不同 Q，卻繼續使用舊 shard。

解法是在 scoped `fork_rng` 中依 `args.seed` 重建 rotation：

- CPU 使用獨立 CPU RNG scope。
- GPU 只 fork 指定 device RNG。
- helper 前後不消耗 caller RNG state。

新增 programmatic resume/RNG drift tests，確保完整轉換與 resume 使用相同 Q。

## 9. 衝突七：accuracy benchmark tokenizer

Incoming low-VRAM accuracy benchmark會先卸載 reference model 再載入 INT4，這是值得保留的功能；但它移除了 ours 的 `--tokenizer-model`，並強制以 reference model 載入 tokenizer。對 gated 或只有本地 checkpoint 的 reference，這會使 benchmark 無法執行。

最終保留兩者：

- sequential low-VRAM reference 流程。
- `--tokenizer-model` override。
- dataset loader 使用 `args.tokenizer_model or args.reference_model`。

## 10. Kernel API 與未使用 prefill 實作

Bindings 合併後同時保留：

- PARD q_len 1–16 KV4 append、native GQA、expanded oracle、legacy/general FFN。
- Incoming K1、multi-scale/interleaved GEMM、grouped-H256 FFN、fused RMSNorm quantization。

`flashinfer/prefill.cuh` 被 incoming 刪除。全 repo 無 include、binding 或 symbol reference，乾淨 build/link 亦通過，因此判定是未使用 standalone implementation，不需為相容性恢復。

## 11. 驗證總結

| Gate | 結果 |
|---|---|
| Merge conflict markers / unmerged index | 無 |
| `git diff --check` | 通過 |
| HIP extension clean compile/link | 通過 |
| Pybind exports | 26/26 存在 |
| 完整 `tests/` | 239 passed |
| 真實 MATH prompt 14，AR/TI，256 tokens | exact |
| 真實 MATH prompt 14，AR/TD，256 tokens | exact |
| Legacy GEMM fallback對舊 formal，前 32 tokens | exact |

Focused merge smoke（非正式 benchmark）：

| Mode | Steady tok/s | 相對 AR | Mean accept |
|---|---:|---:|---:|
| AR | 26.171 | 1.000× | — |
| TI | 28.600 | 1.093× | 3.427 |
| TD | 38.203 | 1.460× | 4.431 |

這些數字只用來驗證整合方向與敏感 prompt；正式跨資料集結果必須用 8 warmups、3 sweeps、HumanEval 80 / GSM8K 80 / MATH-500 20 的既有 contract 重新量測。

## 12. 保留的 fallback 與後續判定

| 開關 | 用途 |
|---|---|
| `QUAROT_FUSED_K1=0/1` | K1 fused RoPE/KV append ablation |
| `QUAROT_FUSED_PROJECTIONS=0/1` | shared projection kernel ablation |
| `QUAROT_GROUPED_SCALE_GEMM=0/1` | 舊 INT32 dequant 與新 FP16 epilogue切換 |
| `QUAROT_PERSISTENT_KV_METADATA=0/1` | persistent metadata ablation |
| `--expanded-mha` | native GQA correctness oracle |

Post-merge 正式 benchmark 若任一 TI/TD dataset 弱於原主線，優先依序檢查：

1. grouped-scale GEMM 是否只加速 AR、卻未同比改善 M=15/16 verifier。
2. fused projections 的 M=1 與 M=15/16 收益是否不對稱。
3. K1 因 legacy parity 被關閉後，AR/transaction 的 kernel 組合是否可改為同一 row-exact writer。
4. persistent metadata 或 cache transaction 是否在 verifier step 引入額外同步。
5. 若新 target rounding 改變 acceptance，需分離「target TPS」與「mean accept」影響，不可只看總 speedup。


## 13. 正式 benchmark 前的 compile／memory 衝突

PARD2 的實際 proposal M 涵蓋 15–30。若在同一個 `StaticCache` 上用 `max-autotune` 一次預編譯全部 shape，cudagraph private pool 與完整 cache buffers 會累積，M=25/26 時 VRAM 接近 30.8 GiB 並 OOM。最終方案是：

- 正式 TI/TD 使用 `max-autotune-no-cudagraphs`。
- 新增 opt-in `--precompile-draft-shapes`，只允許搭配 no-cudagraph mode。
- 每個 M 各自建立短命 `StaticCache`，先做一 token eager prefill，再呼叫 compiled drafter。
- helper 使用 `torch.inference_mode()`，每個 shape 後刪除 cache、GC 並清 allocator。
- 預編譯預設關閉；正式 runner 明確啟用並把 M=15–30 寫入 contract。

空 cache、共用 cache或缺少 inference mode 都曾導致第 17 個 graph 重編譯或 OOM，因此未採用。最終 TI/TD measured sweeps 中沒有 recompilation warning。

## 14. Post-merge 正式結果（目前已完成）

條件為 batch 1、greedy、正常 EOS、256-token 上限、8 warmups、3 sweeps；每個 mode/dataset 為獨立程序。「舊 mode 比」比較同 mode 絕對 steady TPS。

| Dataset | Mode | Steady | E2E | 新 AR speedup | 舊 mode 比 | Mean accept | Steady CV | Peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MATH-500 | AR | 30.005 | 29.698 | 1.000× | 1.422× | — | — | — |
| MATH-500 | TI | 52.688 | 51.591 | 1.756× | 1.228× | 5.787 | 0.253% | 8.53 GiB |
| MATH-500 | TD | 60.209 | 56.937 | 2.007× | 1.264× | 6.489 | 0.761% | 8.60 GiB |
| HumanEval | AR | 30.967 | 30.641 | 1.000× | 1.481× | — | 0.655% | 6.05 GiB |
| HumanEval | TI | 49.783 | 46.391 | 1.608× | 1.058× | 5.656 | 0.215% | 8.62 GiB |
| HumanEval | TD | 57.438 | 53.703 | 1.855× | 1.073× | 6.431 | 0.275% | 8.70 GiB |
| GSM8K | AR | 31.036 | 30.919 | 1.000× | 2.287× | — | 0.640% | 5.97 GiB |
| GSM8K | TI | 49.785 | 47.108 | 1.604× | 1.275× | 5.548 | 2.017% | 8.20 GiB |
| GSM8K | TD | 59.542 | 54.938 | 1.918× | 1.283× | 6.355 | 2.593% | 8.25 GiB |

MATH TI/TD 各 60/60、HumanEval 與 GSM8K TI/TD 各 240/240，均與 post-merge AR exact parity。所有 steady CV 皆低於 5%，VRAM headroom 遠高於 10%。E2E 首 sweep仍含不同 prompt-length kernel 首載，因此採用結論以 steady gate 為主。

### 絕對變快但相對 AR 倍率變小

Post-merge AR 在 MATH/HumanEval 提升約 42%/48%；TI/TD 絕對 TPS 仍較舊 mode 高 6–27%。相對倍率下降是因 K1 等收益主要服務單-token AR，M≈16 verifier 無法直接使用；grouped-scale rounding 也改變 conditional path，使 HumanEval verifier steps 中位數 TI 44→48、TD 38.5→43。

| HumanEval | 舊 verifier/step | 新 verifier/step | 舊 draft/step | 新 draft/step |
|---|---:|---:|---:|---:|
| TI | 103.27 ms | 93.14 ms | 16.32 ms | 11.06 ms |
| TD | 103.78 ms | 92.76 ms | 17.28 ms | 11.09 ms |

因此 speculative stage 本身沒有變慢：verifier/step 約快 10%，draft/step 約快 32%；總倍率下降來自 AR 分母提升更大與 acceptance 改變。

## 15. Acceptance focused ablation

HumanEval prompt 0–7、TI、256-token 上限：

| 配置 | Mean accept | Steady tok/s | current AR parity | 決策 |
|---|---:|---:|---:|---|
| Post-merge 預設 | 5.744 | 53.055 | 8/8 | 保留 |
| 關閉 grouped-scale | 5.907 | 29.075 | 0/8 | 拒絕：−45% TPS且破壞 parity |
| 關閉 fused projections | 5.747 | 38.567 | 8/8 | 拒絕：acceptance 無改善且 −27% TPS |
| eager drafter | 5.758 | 48.325 | 8/8 | 拒絕：compile 不是主因且較慢 |
| 舊原主線 | 6.161 | 51.430 | 舊 AR contract | 歷史參考 |

acceptance 差異主要來自 target execution numerics，而非 drafter compile 或 projection fusion。不能混用舊 target 接受規則與新 target correction，也不能為 acceptance 回退 grouped GEMM。後續應鎖定不改 semantics 的 M=15/16 grouped-scale dispatch、shared metadata/cache transaction/launch overhead；若要恢復 acceptance，需以 current target 為 teacher 重校 drafter。

## 16. Qualification 與採用判定

沿用 `e2e/qualify_pard2.py` 的逐 `(sweep, prompt)` paired bootstrap（10,000 samples）：

| Dataset | TI paired speedup 95% CI | TD paired speedup 95% CI | TI/TD hard gate |
|---|---:|---:|---|
| HumanEval | [1.553, 1.702] | [1.830, 1.901] | pass / pass |
| GSM8K | [1.557, 1.669] | [1.811, 1.982] | pass / pass |
| MATH-500 | [1.565, 2.006] | [1.840, 2.222] | pass / pass |

`all_exact_parity=true`、總 `hard_gate=true`。所有 CI 下界皆大於 1.0，steady run-level CV <5%，VRAM headroom 為 72.7–74.2%。TD literature stretch gate 未通過（HumanEval/GSM8K 約達 Qwen3 參考倍率的 27–30%），故不得宣稱趨近文獻值。

採用結論：保留 post-merge grouped-scale、fused projections、persistent metadata 與 ours PARD2 主線；不採用 grouped-scale/projection/eager 回退。三資料集 AR、TI、TD 的絕對 steady TPS 都高於原主線同 mode，因此沒有接受絕對效能退化。相對新 AR 的 multiplier 低於舊報告，明確列為後續 M=15/16 verifier 與 current-target drafter calibration 的優化工作。
