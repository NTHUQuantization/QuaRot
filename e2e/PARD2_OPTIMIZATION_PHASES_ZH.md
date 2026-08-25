# fused_v1 × PARD2 階段式最佳化紀錄

## 目的與紀錄原則

本文件追蹤 PARD2 在 `fused_v1` 上的四階段最佳化、correctness 證據、效能收益與 trade-off。每一階段均以先前通過 qualification 的實作作為比較基準；只有維持 exact greedy parity 且量測結果穩定的變更，才會成為下一階段的預設基線。

## Phase 1：移除 small-chunk rowwise LM-head（已驗收）

### 動機

先前為了排除 chunk verifier 的數值差異，`exact_small_chunk` 同時啟用了 row-independent RMSNorm 與逐 row 的 LM-head。後續診斷確認 parity 的必要條件是 row-independent RMSNorm；將 M=16 verifier 的 LM-head 拆成 16 次 M=1 呼叫則不是必要條件，而且會引入大量 kernel launch 與 GEMM dispatch 成本。

因此本階段保留 row-independent RMSNorm 的數值契約，僅將 verifier LM-head 恢復為單次 batched small-chunk 呼叫。兩條路徑由獨立開關控制，讓效能回歸時可以單獨 rollback，而不影響 RMSNorm parity 修正。

### 程式設計

- 保留 row-independent HIP RMSNorm，確保每一 row 使用與 sequential M=1 相同的 reduction order。
- 將 small-chunk LM-head 從逐 row 執行改為 batched 執行，避免最多 16 次 M=1 dispatch。
- 保留 rowwise LM-head rollback flag，必要時可回到舊路徑，並用於 A/B correctness 與效能比較。
- AR、PARD2-TI、PARD2-TD 仍共用相同的 target、cache 與 verifier 數值契約；本階段不改 drafter、acceptance rule 或 KV transaction。

### Correctness 證據

- 已知敏感案例 MATH prompt 14、chunk length 16：batched LM-head 相對 rowwise 路徑的完整 logits exact，KV cache exact。
- regression suite：117 passed。
- paired smoke benchmark：9/9 輸出保持 exact token parity。
- 平均接受長度完全不變：4.48276 → 4.48276。

### Paired smoke 設定

- 模式：PARD2-TI。
- 資料：MATH prompts 14–16。
- 每個 prompt 生成 128 tokens。
- 2 次 warmups。
- 3 次 measured sweeps。
- 比較方式：相同 prompts、checkpoint、draft_k 與 runtime 設定，只切換 rowwise／batched LM-head。

### 效能結果

| 指標 | Rowwise LM-head | Batched LM-head | 變化 |
|---|---:|---:|---:|
| Steady decode | 20.8066 tok/s | 27.4252 tok/s | **+31.81%** |
| E2E throughput | 20.5926 tok/s | 26.9288 tok/s | **+30.77%** |
| Verify median | 4099.06 ms | 3041.05 ms | −1058.01 ms（−25.81%） |
| Mean accept length | 4.48276 | 4.48276 | 無變化 |
| Token parity | 9/9 | 9/9 | exact |

### 效益與 trade-off

效益：在不改變接受長度與輸出 token 的前提下，單次 batched LM-head 移除了逐 row launch/dispatch overhead，使 paired smoke 的 steady 與 E2E throughput 分別提升 31.81% 與 30.77%。verify median 同時下降約 25.81%，證明收益直接來自 verifier 路徑，而非 acceptance 波動。

Trade-off：batched GEMM 的累加方式理論上可能與逐 row GEMM 不同，因此不能只依賴 top-1 parity；每次調整 LM-head kernel、shape dispatch 或工具鏈後，都必須重跑完整 logits、KV exact 與端到端 token parity。為降低風險，rowwise LM-head 保留為獨立 rollback/oracle flag，但不再作為預設路徑。

### 階段結論

Phase 1 已通過 correctness 與 paired smoke 驗收，batched LM-head 成為後續階段的效能基線。row-independent RMSNorm 繼續保留；其本體成本與後續 norm/quant fusion 將於 Phase 4 評估。

## Phase 2：TD runtime 與 feature path 最佳化（已驗收）

### 動機與候選方案

TD 相較 TI 具有較高接受長度，但 feature basis restore、hidden-row 處理與 target projection 可能抵銷 acceptance 優勢。本階段依序評估三項不改變 target 計算的 runtime 最佳化：

1. 將 inverse-Hadamard 所需的 rotation signs 與 final RMSNorm gamma 預先快取於 GPU（cached basis）。
2. acceptance 決定後才 restore 實際需要的 hidden rows（lazy features）。
3. target projection 只計算 unique feature rows，再於 projected space 複製 mask rows（unique-row projection）。

三項均保留獨立旗標，方便 paired ablation 與 rollback；只有通過 token parity、acceptance 與穩定性 gate 的組合才會成為預設。

### Paired smoke 設定

- 模式：PARD2-TD。
- 資料：MATH prompts 14–16。
- 每個 prompt 生成 128 tokens。
- 2 次 warmups。
- 3 次 measured sweeps。
- 比較基線：已採用 Phase 1 batched LM-head 的 TD runtime。
- Phase 1 基線 aggregate：steady 31.2128 tok/s、E2E 30.5418 tok/s、mean accept length 4.7778。

### Correctness 證據

- cached basis、cached + lazy、cached + lazy + unique 三個組合均為 9/9 exact token parity。
- 所有組合的 mean accept length 均保持 4.7778，沒有 acceptance 回歸。
- 完整 regression suite：120 passed。

### 結果與採用決策

| 配置 | Steady paired median | E2E paired median | 其他觀察 | 決策 |
|---|---:|---:|---|---|
| Phase 1 baseline | 1.00000× | 1.00000× | 31.2128 / 30.5418 tok/s | 基線 |
| Cached basis | **1.05357×** | **1.04857×** | aggregate 35.1243 / 29.8592 tok/s；CV 2.95% | **採用** |
| Cached + lazy | 1.02357× | 1.01727× | 相對 cached-only 回歸 | 不採用 |
| Cached + lazy + unique | 0.94641× | 0.94497× | warmed draft median 617.42 ms，基線 539.13 ms | 不採用 |

Cached basis 三個 sweep 的 paired steady ratios 為 `[1.05265, 1.04857, 0.99792]`。雖然第三個 sweep 接近持平，整體 paired median steady 與 E2E 仍分別提升 5.357% 與 4.857%，且 sweep CV 為 2.95%，低於 5% gate。aggregate E2E 的 29.8592 tok/s 低於基線 aggregate 30.5418 tok/s，與 paired 結論不一致；因此採用判斷以控制相同 sweep 條件的 paired ratios 為主，而不以跨執行 aggregate 直接定論。

### 效益與 trade-off

Cached basis 消除了每輪 signs/gamma 的重複 device/dtype materialization，在不改變 token 或 acceptance 的前提下取得約 5% paired throughput 收益。此路徑已成為預設，並保留 `--no-td-cache-basis` 作為 rollback／correctness oracle。

Lazy features 在單獨對 Phase 1 基線比較時仍有小幅正收益，但低於 cached-only，表示減少 restore rows 的理論節省被 slicing、控制流程或 shape 變化成本抵銷。Unique-row projection 進一步造成負收益；其 warmed draft median 由 539.13 ms 增至 617.42 ms，顯示動態 row shape／compile signature 與較差的小 M 執行效率超過省下的 projection FLOPs。

因此 lazy features 與 unique-row projection 不設為預設，其實驗旗標保留且預設為 false，供後續固定 shape、graph capture 或 kernel fusion 研究使用。Phase 3 以 cached basis 為新基線，避免把兩個負向 ablation 帶入後續結果。

## Phase 3：TD projection basis folding（已完成實驗，未採用）

### 動機與設計

Legacy TD feature path 會對四個 4096-wide target taps 逐一執行 inverse Hadamard 與 rotation-sign restore；final tap 另需恢復原始 RMSNorm gamma，然後才串接成 16384 維輸入官方 `target_proj`。本階段評估把這些線性 basis restore 直接折入 projection weight，以消除 decode loop 中的四組 restore 工作。

在 row-vector convention 下，QuaRot activation 為 `x_rot = x D H`。官方 projection weight 依四個 tap 分為 `W0..W3`，啟動時在 CPU FP32 做 blockwise transform：

```text
W0' = W0 Γ D H
Wi' = Wi D H, i = 1, 2, 3
```

實作沿 weight 最後一維執行 normalized Hadamard，不建立 dense H；完成後再轉回 drafter dtype。folded collector 直接使用 target final norm 的 rotated-normalized 輸出作第一個 tap，另外三個 tap 保留 layer 28/20/12 的 raw rotated 輸出。官方 checkpoint 不遷移、不覆寫，轉換只在 runtime 啟動時作用於記憶體中的 projection 副本，並設有 double-fold guard。

`--td-basis-fold` 是 experimental opt-in／rollback flag，預設維持 false；legacy collector/projection 仍是 correctness oracle。Raw per-channel calibration 的 affine scale/bias 尚未代數折入 weight，因此 raw calibration 與 basis fold 同時啟用時會明確拒絕。Projected calibration 位於 projection 後，不受此限制。

### Correctness 證據

- focused regression suite：127 passed。
- eager、MATH prompt 14、32 generated tokens：folded 與 legacy output tokens exact parity。
- eager 診斷的 mean accept length 由 4.5714 降至 4.0，顯示 BF16 cast 與新的運算順序可能造成 drafter candidate 漂移；即使最終 greedy output 相同，也不能把 output parity 當作 candidate parity。
- max-autotune paired smoke：9/9 output exact parity。
- max-autotune 兩路 mean accept length 皆為 5.95254，沒有 acceptance 回歸。

### Max-autotune paired smoke 設定

- 模式：PARD2-TD。
- 資料：MATH prompts 14–16。
- 每個 prompt 生成 128 tokens。
- 2 次 warmups。
- 3 次 measured sweeps。
- 基線：Phase 2 cached-basis 預設。
- 結果檔：
  - `phase3/td_math_phase2_baseline_rerun.json`
  - `phase3/td_math_basis_fold.json`

Sweep 0 仍包含首次動態 shape compilation，不能代表 warmed steady state，因此效能採用判斷只使用 sweep 1–2 的 paired samples；原始三個 sweeps 仍完整保留於結果檔，不刪除或改寫。

### 效能結果

| 指標 | Basis fold / Phase 2 baseline | 採用門檻 | 結果 |
|---|---:|---:|---|
| Paired median steady | 1.00319× | ≥1.03× | 未通過 |
| Paired median E2E | 1.00491× | ≥1.03× | 未通過 |
| Warmed steady CV | 約 4.46% | <5% | 通過 |
| Warmed E2E CV | 約 4.44% | <5% | 通過 |
| Mean accept length | 5.95254 → 5.95254 | 不回歸 | 通過 |
| Token parity | 9/9 exact | 100% | 通過 |
| Peak VRAM | 約下降 0.2 MiB | 不回歸 | 通過 |

### 效益與 trade-off

Basis folding 成功移除 runtime 的 inverse-Hadamard/sign/gamma feature restore，並略降約 0.2 MiB peak VRAM；correctness、正式 smoke acceptance 與 warmed-run CV 均通過。然而 paired median steady 與 E2E 只提升約 0.319% 與 0.491%，遠低於 3% 採用 gate，說明原 feature restore 在目前整體 step 中並非主要瓶頸，或節省被 projection/collector/launch 的其他成本掩蓋。

主要 trade-off 是啟動時增加 CPU FP32 weight folding 成本與額外記憶體峰值，而且 BF16 cast／運算順序可能改變 drafter candidates。Eager 小樣本已觀察到 accept length 下降，雖然 max-autotune 正式 smoke 沒有重現，因此未來改變 PyTorch、ROCm、projection kernel 或 dtype 時仍需重跑 candidate agreement 與 acceptance gate。Raw calibration 尚不相容也是目前限制。

### 階段結論

Phase 3 的 algebra、實作與 correctness 已驗證，但效能未達 3% gate，因此 `td_basis_fold` 維持 default false，只保留為 experimental ablation／rollback 路徑。Phase 4 不以 basis fold 作為新基線，繼續使用 Phase 2 cached-basis 預設。

## Phase 4：exact norm-quant 與 hidden-tap hook 評估（已完成，未採用）

### Norm–INT4 fusion 設計

本階段把 row-independent RMSNorm 與其後緊接的 activation INT4 quantization 合併，避免逐 row norm 結果寫回 global memory、再由量化路徑重新讀取。融合路徑維持既有 exact 數值契約：每一 row 的 FP32 reduction 順序、FP16 normalization rounding point、`max(abs(x))/7` scale、round/clamp、packed INT4 layout 與 scale dtype 都不改變；attention 的 QKV 與 MLP 的 gate/up consumer 可直接接收 fused packed tensor。最終 LM-head 前的 model norm 不接 INT4 consumer，因此仍走原 exact norm 路徑。

實作由獨立 experimental flag 控制，預設為 false。這讓收益可明確歸因於 norm–quant fusion，並保留原 RMSNorm + quantizer 作 bit-exact oracle；不需要 checkpoint migration，也不改 acceptance rule、KV transaction 或 drafter。

### Correctness 證據

- fused norm–quant 的 packed INT4 bytes 與 scales 對 legacy 路徑維持 bit parity。
- focused regression suite：**148 passed**。
- AR eager A/B：exact greedy token parity。
- TI eager A/B：3/3 exact token parity，mean accept length **3.61111 → 3.61111**。
- TD eager A/B：output tokens exact，accept length 不變。
- TD max-autotune paired smoke：9/9 exact token parity；baseline 與 fusion 的 mean accept length 均為 **5.95254**。

### AR、TI 與 TD eager 診斷

| 模式 | Steady ratio | E2E ratio | Correctness | 解讀 |
|---|---:|---:|---|---|
| AR eager | **1.10637×** | **1.03397×** | exact tokens | 單 token AR 可直接受益；E2E 收益受 prefill／固定成本稀釋 |
| TI eager | **1.04896×** | **1.05770×** | 3/3 exact tokens；accept 3.61111 不變 | 通過 3% smoke gate |
| TD eager | **1.02265×** | **0.99434×** | exact tokens／accept | verifier target 小幅變快，但完整 TD step 略為負收益 |

TI eager 使用 MATH prompt 14、64 generated tokens、1 次 warmup 與 3 次 measured sweeps。Paired steady／E2E CV 分別為 **0.27%／0.79%**；三次 target verify 由約 1770.7／1768.4／1793.9 ms 降至 1678.7／1664.3／1700.0 ms，顯示 TI 的收益穩定且來自 verifier。

Eager 結果顯示融合確實消除一部分 row-independent norm/quant overhead；然而 TD 的 draft、projection、acceptance 與其他 runtime 成本會稀釋 target 改善，因此不能由 AR 或 target-only 結果推定 TD E2E 採用價值。TI 雖通過 3% smoke gate，但 AR steady 收益更大（10.637%），而 TD 正式 smoke 只有 1.333% E2E；若把 fusion 設為共同 target 預設，會縮小 speculative 相對同 target AR 的公平 speedup。TI 也尚未完成三資料集 formal 驗證，因此本階段不採用全域 default，只保留 opt-in 與後續 mode-specific 研究；正式比較不使用不同 target 配置。

### TD max-autotune paired smoke

- 模式：PARD2-TD，MATH prompts 14–16。
- 每個 prompt 生成 128 tokens，2 次 warmups、3 次 measured sweeps。
- 比較基線：Phase 3 使用的 Phase 2 cached-basis baseline；`td_basis_fold` 不啟用。
- Sweep 0 含首次動態 shape compilation，因此採用判斷排除 sweep 0、使用 sweep 1–2；原始資料仍完整保留。

| 指標 | Fused norm–quant / Phase 3 baseline | 採用門檻 | 結果 |
|---|---:|---:|---|
| Paired median steady | **1.01031×** | ≥1.03× | 未通過 |
| Paired median E2E | **1.01333×** | ≥1.03× | 未通過 |
| Warmed steady CV | **4.72%** | <5% | 通過 |
| Warmed E2E CV | **4.45%** | <5% | 通過 |
| Target verify median | 2892.12 → **2860.34 ms** | 改善 | −31.78 ms（−1.10%） |
| Mean accept length | 5.95254 → **5.95254** | 不回歸 | 通過 |
| Token parity | **9/9 exact** | 100% | 通過 |
| Peak VRAM | 8,840,159,232 → **8,839,307,264 bytes** | 不回歸 | 約下降 0.81 MiB |

Fusion 維持 parity、acceptance 與 VRAM gate，target verify median 也確實下降；但 warmed paired median steady／E2E 只提升 1.031%／1.333%，低於 3% 採用門檻。因此 fused norm–quant flag 維持 default false。其 trade-off 是增加一套跨 RMSNorm、quantizer 與 packed-tensor consumer 的專用契約及測試面，且收益依賴 Qwen3 shape、ROCm kernel dispatch 與 verifier 工作比例；目前效益不足以抵銷維護成本。

### Folded hidden-tap hook profiler

為判斷 Phase 3 basis folding 後是否值得把 hidden taps 融入 target kernel，另以同一已載入 fused target、固定 input/cache shape，交錯量測無 hook、四個 no-op hook、只保存 tensor reference 的 folded `SelectedHiddenCollector` hook。每次呼叫均以 cache transaction rollback 回到相同 logical length，量測前後同步 GPU，q_len 1 與 16 各取得 30 samples。

| Shape | Store-ref 相對 no-hook median | Correctness | 結論 |
|---|---:|---|---|
| q_len=1 | **−0.293%** | logits／top-1 exact | 負值屬量測 noise |
| q_len=16 | **+0.0919%** | logits／top-1 exact | 可忽略 |

兩個 shape 的 store-ref hook overhead 均低於 0.5%，遠低於 3% E2E gate。這表示目前 TD 差距不在 Python hook callback 或保存 hidden tensor reference；為移除不到 0.5% 的成本改成 wrapper、explicit model forward 或 HIP side-write/fusion，會增加 transformers 結構耦合、維護與 correctness 風險，卻沒有可觀收益。因此不繼續實作 hidden-tap target fusion。

### Wave-tail reduction 決策

原先候選的 wave-tail reduction-tree 微調沒有實作。若 fused norm–quant 不採用，legacy 路徑每個 verifier forward 仍有 72 個 layer norms，因此理論上仍可獨立研究 reduction tail；但其預估收益低於已完成的 norm–quant fusion。較直接、可一次消除 intermediate write/read 的 fusion 尚且只取得 1.333% TD E2E 收益，已未通過 gate，再投入 architecture-specific wave/barrier ordering 最佳化更不值得，且會提高 gfx1201 專屬維護與數值順序風險。

### 結果檔

- `pard2_optimization_results/phase4/ar_eager_baseline.json`
- `pard2_optimization_results/phase4/ar_eager_fused_norm_quant.json`
- `pard2_optimization_results/phase4/ti_eager_baseline.json`
- `pard2_optimization_results/phase4/ti_eager_fused_norm_quant.json`
- `pard2_optimization_results/phase4/td_eager_fused_norm_quant.json`
- `pard2_optimization_results/phase4/td_math_fused_norm_quant.json`
- `pard2_optimization_results/phase4/td_hook_overhead_30.json`
- Phase 3 paired baseline：`pard2_optimization_results/phase3/td_math_phase2_baseline_rerun.json`

### 階段結論

Phase 4 已完成 norm–quant、hook overhead 與後續 target-fusion 可行性評估。Norm–quant 在 AR eager 有明顯局部收益，TI eager 亦通過 3% smoke gate，且皆維持 bit/token/acceptance parity；但正式 TD max-autotune E2E 只提升 1.333%，未達 gate，而共同 target 下 AR 收益更大。為維持 AR／TI／TD 公平比較且避免以單 prompt 取代 formal qualification，fusion 不升為全域預設，只保留 experimental opt-in 與 mode-specific 後續研究。Hidden store-ref hook overhead 低於 0.5%，因此 hidden-tap fusion 與 wave-tail reduction 不繼續實作；後續效能基線仍為 Phase 2 cached-basis 配置。

## 最終整合與 qualification recheck

### 完整 regression

最終整合後執行完整 `pytest -q tests`，結果為 **161 passed、4 warnings**。Warnings 為既有 Triton/runtime 警告，沒有測試失敗；Phase 1–4 的程式路徑、rollback flags、cache/KV correctness、模型相容性與 PARD2 runtime 測試均通過。

### 既有正式結果重新驗證

`pard2_optimization_results/qualification_phase1_recheck.json` 重新分析先前已完成的正式 HumanEval 80、GSM8K 80、MATH-500 20 結果；此步是 qualification 邏輯 recheck，不是以新的 smoke run 取代正式資料。彙總結果為：

- `all_exact_parity=true`
- `hard_gate=true`
- 所有 TI／TD dataset 的 steady speedup 95% bootstrap CI 下界均大於 1.0。
- 所有 run-level CV 均小於 5%。

| Dataset | Mode | Median speedup | 95% CI 下界 | Run-level CV | Exact parity／hard gate |
|---|---|---:|---:|---:|---|
| HumanEval 80 | TI | 1.66680× | 1.62040× | 0.936% | true／true |
| HumanEval 80 | TD | 1.92115× | 1.84255× | 0.580% | true／true |
| GSM8K 80 | TI | 2.36069× | 2.14960× | 0.226% | true／true |
| GSM8K 80 | TD | 2.80185× | 2.55858× | 0.499% | true／true |
| MATH-500 20 | TI | 1.58649× | 1.47930× | 1.098% | true／true |
| MATH-500 20 | TD | 1.74305× | 1.70047× | 0.338% | true／true |

`td_stretch_gate=false` 仍維持不變：目前相對文獻 speedup／accept-length 的延伸目標尚未達成。Stretch gate 不是此次四階段架構最佳化的採用門檻，也不推翻 exact parity、CI、CV、VRAM 所構成的第一階段 hard gate 結論。

### 最終預設與未採用項目

最終預設只納入兩項已通過 correctness、穩定性與效能 gate 的變更：

1. **Phase 1 batched LM-head**：保留 exact row-independent RMSNorm，但 small-chunk LM-head 不再逐 row dispatch；rowwise LM-head 預設為 false。
2. **Phase 2 cached TD basis**：rotation signs 與 final RMSNorm gamma 預先快取於 target GPU；`td_cache_basis` 預設為 true。

其餘候選保留為可獨立 rollback／ablation 的 experimental opt-in，未改變正式預設：

- Phase 3 `td_basis_fold=false`：correctness 通過，但 warmed E2E 只提升約 0.491%，未達 3% gate。
- Phase 4 fused norm–quant 預設為 false：TI eager smoke 雖通過 3%，但 TD max-autotune E2E 只提升 1.333%，且共同 target 下 AR 收益更大；尚不足以作為公平的全域預設。
- Hidden-tap target fusion 不實作：30-sample profiler 顯示 store-ref hook overhead 小於 0.5%，不值得增加 target/HIP 結構耦合。
- Wave-tail reduction 不實作：預估收益低於已未通過 TD gate 的 norm–quant fusion，architecture-specific 維護與數值風險不成比例。

因此四階段工作完成後，正式基線仍維持「Phase 1 batched LM-head + Phase 2 cached basis」；Phase 3／4 的實驗程式與結果則保留作為後續 mode-specific 最佳化與 correctness oracle。
