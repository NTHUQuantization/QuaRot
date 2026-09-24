# Qwen3-32B × PARD2 TD：S3–S5 最終實驗紀錄

日期：2026-08-28（Asia/Taipei）
狀態：S3–S5 完成；S8 1-prompt cache tier 通過，但 expanded HumanEval gate 失敗，未進 S9。
部署建議：**維持既有 TI**。S3 只保留為獨立 held-out winner，S5 為探索候選；兩者均不得升為 canonical TD。

## 1. 結論摘要

在不修改 QuaRot 或 PARD2 架構的前提下，已完成 `smoke_train_r2` 後續除錯、S3、S4、S5、獨立 held-out scorer，以及 HumanEval／GSM8K／MATH-500 runtime smoke。

| 候選 | Validation loss | Mean acceptance | Median E2E tok/s | 相對基線 | Gate |
|---|---:|---:|---:|---:|---|
| TI | — | 4.0680 | 12.2239 | — | 已有基線 |
| S3, alpha=0.25 | 3.3607 | 4.8810 | 13.8159 | paired E2E / parent = 1.0232 | **PASS；保守 winner** |
| S4, alpha=0.05 | 2.8517（full） | 4.8235 | 13.2598 | paired E2E / S3 = 0.9662 | **FAIL parent gate** |
| S5, full | 3.1855 | **4.9540** | **14.4194** | paired E2E / TI = 1.1062；/ S3 = 0.9801 | PASS vs TI；探索候選 |

S5 的 mean acceptance 比 S3 高約 1.50%，但同 prompt paired E2E 比 S3 低約 1.99%。因此目前沒有充分理由取代 S3；S5 應進入較大樣本 scorer 或完整 benchmark 後再決定。

## 2. 凍結設定與資料

- Target：Qwen3-32B GPTQ W4A4KV4 fused_v1。
- Drafter：AMD PARD2-Qwen3-14B，PARD-2 draft length K=16。
- Projection：`20480 → 1024`；QuaRot、PARD2 inference/training 拓撲均未改動。
- HIP kernel：`quarot_HIP_gfx1201_g4_m2.so`，SHA256 `8e544408498612b2eae27b6fce9a52e939bded735eba77772692f6aa5fe5cfc5`。
- Runtime：grouped waves 4、multi waves 2、eager attention、`QUAROT_FUSED_K1=0`。
- Train：2,279 samples、1,000,461 source tokens；general/code/math 各約 333K tokens。
- Validation：12 samples、5,577 tokens，每個 stratum 4 samples。
- Train、validation 與 formal held-out overlap 均為 0。
- Teacher cache 為離線 exact BF16 selected features + FP32 gold probabilities；loss 使用 frozen 32B LM head 做 exact full-vocabulary chunked CE+KD，沒有 top-k 近似。

## 3. `smoke_train_r2` 至穩定 trainer 的除錯鏈

1. `smoke_train_r1` 的完整 row expansion 在 validation forward 達 31,105 MiB／95.34%，觸發 VRAM hard gate。
2. r2/r3 加入 no-grad validation、gradient checkpointing 與 vocabulary chunking，仍無法消除 expanded-row activation 峰值。
3. r4–r7 改為最多 128-token training window，VRAM 已降至安全範圍，但 draft forward 出現 ROCm exception。
4. 根因是強制 FlashAttention2 與 PARD2 的 4D custom attention mask 不相容；切回 Transformers eager attention 後，r8 成功，峰值 3,970 MiB。
5. 首次正式 S3 使用 BF16 AdamW，於 optimizer step 106 發生 projection 與 Adam state NaN。改用 FP32 master projection、FP32 `exp_avg`／`exp_avg_sq`，forward 才 cast BF16，數值穩定。
6. LR `3e-5` 在固定樣本位置仍有 non-finite gradient；LR `1e-5` 加入 finite guard、官方 accumulation 語意及 final pending optimizer step 後，完成 S3。

這些變更均屬記憶體排程、attention backend 與 optimizer numeric precision 修正，沒有更動 QuaRot／PARD2 架構、K、COD、taps、loss 定義或 inference protocol。

## 4. S3：projection micro

穩定正式訓練結果：

- Source tokens：128,355；實際 trained tokens：114,876。
- Throughput：297.0 trained tok/s。
- Validation loss：3.652401 → 2.869796。
- Peak VRAM：4,250 MiB／13.03%；hotspot/memory：93°C／74°C。

直接使用完整 delta 在 held-out 上退化，因此執行 projection delta shrink。`alpha=0.25` 勝出：

- 獨立 validation loss：3.360734。
- Mean acceptance：4.880952。
- code/general/math acceptance：4.7500／3.2195／9.6667。
- Median E2E／steady：13.8159／18.2129 tok/s。
- 相對 frozen parent 的 paired median E2E：1.0232×。
- AR output exact parity：通過。

S3 gate：**PASS**。

## 5. S4：projection medium

由 S3 winner 開始、reset optimizer、LR `3e-6`，累積訓練至 500,321 source tokens：

| Chunk | Cumulative source tokens | Trained tokens | Validation loss | Throughput |
|---|---:|---:|---:|---:|
| 1 | 250,241 | 220,605 | 3.360734 → 3.026382 | 321.95 tok/s |
| 2 | 375,398 | 330,376 | 3.026382 → 2.909334 | 312.66 tok/s |
| 3 | 500,321 | 439,758 | 2.909334 → 2.851693 | 310.48 tok/s |

- Training peak VRAM：最高 4,390 MiB／13.46%。
- Teacher extraction peak VRAM：27,678 MiB／84.84%；最高 hotspot/memory 103°C／80°C。
- Full delta held-out mean acceptance：4.612903。
- Shrink sweep 最佳非零候選為 `alpha=0.05`：mean acceptance 4.823529，median E2E 13.259847 tok/s。
- `alpha=0.05` 相對 S3 paired E2E：0.966169×；general acceptance 亦下降約 2.38%。

S4 parent no-regression gate：**FAIL**。Validation loss 的改善沒有轉化成 held-out speculative decoding 改善。

## 6. S5：projection full（探索）

為避免沿用 S4 已顯示退化的 projection，權重回到 S3 winner；保留 500K scheduler position、reset optimizer，以 LR `1e-6` 訓練未見過的後半資料至完整 1M source tokens。

| Chunk | Cumulative source tokens | Trained tokens | Validation loss |
|---|---:|---:|---:|
| 1 | 625,383 | 545,726 | 3.360734 → 3.274867 |
| 2 | 750,113 | 654,440 | 3.274867 → 3.226274 |
| 3 | 875,496 | 765,080 | 3.226274 → 3.200633 |
| 4 | 1,000,461 | 875,050 | 3.200633 → 3.185546 |

- Final optimizer step：908；global unit：7,229。
- Final training peak VRAM：4,171 MiB／12.79%；hotspot/memory 94°C／76°C。
- Teacher extraction peak range：27,678–28,708 MiB／84.84–88.00%；最高 hotspot/memory 103°C／84°C。
- Projection optimizer SHA256：`3108c897...e59f`；deployed warp SHA256：`be15df1d...77174`（完整值保存在 checkpoint／manifest）。

Held-out：

- Mean acceptance：4.954023。
- code/general/math acceptance：4.7500／3.2195／9.2222。
- Median E2E／steady：14.4194／18.4767 tok/s。
- Paired median E2E / TI：1.106202×。
- Paired E2E / S3：0.980149×。
- 各 stratum acceptance / TI：code 1.3053×、general 1.1707×、math 1.1232×。
- AR output exact parity：通過。

S5 通過其 TI gate，但相對 S3 有約 1.99% paired E2E regression，故標記為 **exploratory**，不直接覆蓋保守 deployment winner。

## 7. 三 benchmark runtime smoke 與 VRAM

S5 full checkpoint 各跑 1 prompt、32 generated tokens、2K cache、1 warmup／1 sweep。三者皆通過 kernel SHA 與 runtime safety gate；此規模只驗證可執行性與 VRAM，不是 accuracy 或 speedup qualification。

| Benchmark | Mean acceptance | E2E tok/s | Steady tok/s | Peak VRAM | Qualified |
|---|---:|---:|---:|---:|---|
| HumanEval | 1.0000 | 3.5475 | 3.6644 | 28,259 MiB／86.62% | No |
| GSM8K | 1.0000 | 3.5597 | 3.6714 | 28,259 MiB／86.62% | No |
| MATH-500 | 1.0000 | 3.6558 | 3.7461 | 28,259 MiB／86.62% | No |

三個 offset-0 prompt 都沒有 accepted draft token，因此不能用這組 smoke 推論 TD speedup。下一個合理步驟是先擴大獨立 scorer，再以凍結的 S3 與 S5 做完整 80／80／20 × 3 sweeps；8K gate 維持使用者指定的 95%。

## 8. Artifact 索引

```text
qwen3_32b_results/td_calibration/S3_S5_FINAL_RESULTS.json
qwen3_32b_results/td_calibration/S3_projection_micro/formal_fp32master_lr1e5/train_result.json
qwen3_32b_results/td_calibration/S3_projection_micro/shrink_alpha_0.25/
qwen3_32b_results/td_calibration/S3_projection_micro/shrink_alpha_0.25/heldout/td.json
qwen3_32b_results/td_calibration/S4_projection_medium/
qwen3_32b_results/td_calibration/S4_projection_medium/shrink_from_s3_alpha_0.05/heldout/td.json
qwen3_32b_results/td_calibration/S4_projection_medium/heldout_ti/ti.json
qwen3_32b_results/td_calibration/S5_projection_full/base_s3winner_at_500k/candidate.json
qwen3_32b_results/td_calibration/S5_projection_full/chunk_04_train/projection_optimizer.pt
qwen3_32b_results/td_calibration/S5_projection_full/heldout_full/td.json
qwen3_32b_results/td_calibration/S5_projection_full/benchmark_smoke/pard2_td_humaneval.json
qwen3_32b_results/td_calibration/S5_projection_full/benchmark_smoke/pard2_td_gsm8k.json
qwen3_32b_results/td_calibration/S5_projection_full/benchmark_smoke/pard2_td_math_500.json
```

為控制磁碟用量，各 chunk 在 checkpoint／manifest 驗證後刪除了可重建的 teacher `.pt` shards；資料 manifest、來源、程式、checkpoint 與 SHA 記錄均保留。若要重跑 loss，可由 frozen source ranges 重建 teacher shards。

## 9. 決策

- **獨立 held-out 保守候選：S3 alpha=0.25。** S8 expanded failure 後不可直接升為 canonical TD。
- **研究候選：S5 full。** Acceptance 較高且明確勝過 TI，但尚未勝過 S3 的 paired E2E。
- **淘汰：S4。** 不因 validation loss 較低而放寬已凍結的 parent gate。
- 下一輪不得再用單一 prompt smoke 作 speedup 判斷；應對 S3、S5 執行相同 prompt、相同 seed、相同 sweep 的 paired benchmark。

PARD-2 官方實作基準：[AMD-AGI/PARD](https://github.com/AMD-AGI/PARD)。

## 10. S8：2K／4K／8K cache 與 expanded gate

### 10.1 1-prompt cache tier

S5 full checkpoint、每個 benchmark 1 prompt、32 generated tokens；唯一變因為 cache length。

| Cache | HumanEval peak | GSM8K peak | MATH-500 peak | Exact parity | Gate |
|---:|---:|---:|---:|---|---|
| 2K | 28,259 MiB／86.62% | 28,259 MiB／86.62% | 28,259 MiB／86.62% | baseline | PASS |
| 4K | 28,403 MiB／87.06% | 28,403 MiB／87.06% | 28,402 MiB／87.06% | 3/3 vs 2K | PASS |
| 8K | 28,675 MiB／87.90% | 28,675 MiB／87.90% | 28,675 MiB／87.90% | 3/3 vs 2K/4K | **PASS** |

8K 最高 hotspot 64°C；95% VRAM gate 保有約 7.1 percentage points 餘裕。三個 offset-0 prompt 的 mean acceptance 均為 1.0，因此此 tier 只證明 cache safety／parity，不證明 speedup。

### 10.2 Expanded HumanEval gate

接著依 frozen S8 matrix 跑 8K、16 HumanEval prompts、256 generated tokens、1 sweep，並與既有同 target／同 HIP SHA 的 TI formal artifact 中相同 prompts、sweep 0 做 paired 比較：

- Output exact parity：**16/16**。
- S5 TD mean acceptance：**1.000492**。
- Accepted／proposed draft tokens：**2／61,410**。
- TD median E2E：3.7191 tok/s；TI subset median E2E：21.9395 tok/s。
- Paired median TD/TI E2E：**0.168762×**（range 0.1037–0.2239×），遠低於 1.03 gate。
- Peak VRAM：28,675 MiB／87.90%；hotspot/memory：91°C／78°C；資源 gate 通過。

因此 S8 expanded gate：**FAIL**。依預先凍結的 per-task gate，在 GSM8K 載入、尚未 decode 時停止其 workload，MATH-500 未啟動；不以其他 task 平均掩蓋 HumanEval failure，也不進 S9。

這個結果把問題範圍縮小為 **prompt/template distribution alignment**：同一 checkpoint 在獨立 held-out code prompts 的 acceptance 為 4.75，但正式 HumanEval 前 16 prompts 幾乎零接受；所有輸出仍與 TI/AR exact parity，8K VRAM 也安全。因此根因不是 QuaRot kernel、cache size 或 correctness，而是現有 projection calibration／training distribution 沒有覆蓋正式 benchmark 的 draft conditional distribution。

進一步證據顯示兩條路徑都使用 `You are a helpful assistant.` 與 `enable_thinking=False`，且 HumanEval input 76–277 tokens 落在 held-out code 的 34–512 範圍，故不是 thinking flag 或長度區間錯誤。主要差異是 calibration/validation code 來自 Magicoder Evol-Instruct 的完整 teacher-forced responses，而正式 HumanEval 是短函式 completion 並沿 target 自生成 prefix rollout；目前最合理的根因是內容型態與 rollout state distribution shift。

下一步應先建立 benchmark-style、但與正式 80/80/20 零重疊的 scorer/calibration prompts，檢查 ChatML template、thinking prefix、prompt length 與各 draft position 的 conditional acceptance；在此之前不建議直接進 S6 top-4 unfreeze，也不應跑 S9。

新增 artifacts：

```text
qwen3_32b_results/td_calibration/S8_cache_smoke/s5_full/S8_ONE_PROMPT_SUMMARY.json
qwen3_32b_results/td_calibration/S8_cache_smoke/s5_full/S8_EXPANDED_GATE.json
qwen3_32b_results/td_calibration/S8_cache_smoke/s5_full/S8_DISTRIBUTION_SHIFT_EVIDENCE.json
qwen3_32b_results/td_calibration/S8_cache_smoke/s5_full/cache4k/
qwen3_32b_results/td_calibration/S8_cache_smoke/s5_full/cache8k/
qwen3_32b_results/td_calibration/S8_cache_smoke/s5_full/heldout_16_16_8/pard2_td_humaneval_cache8192_tok256_limit16_off0.json
```

## 11. S10：zero-overlap code-rollout alignment

S8 HumanEval failure 後，從 `google-research-datasets/mbpp` full/train 建立 HumanEval-style function stubs。資料固定於 revision `4bb6404fdc6cacfda99d4ac4205087b89d32030c`、CC-BY-4.0；與正式 HumanEval 80 prompts 的 normalized exact、token hash、長度至少80的substring overlap均為0。

- Rollout train：64 prompts、10,258 prompt tokens；由部署Qwen3-32B deterministic AR各生成96 tokens。
- Frozen sequences：16,402 total tokens，其中6,144 target-generated rollout tokens。
- Scorer：16個互斥MBPP prompts、2,607 prompt tokens；AR各生成32 tokens。
- Train/scorer task-ID overlap：0。
- Target rollout generation peak：27,056 MiB／82.93%，最高96/82°C。
- Train teacher extraction peak：28,156 MiB／86.30%；validation teacher peak 27,731 MiB／85.00%。

新 scorer 成功重現 formal failure，且不使用正式 benchmark 調參：S3 exact parity 16/16，但 mean acceptance 1.003922，7,650 proposals只接受2 tokens，paired TD/AR E2E 0.247630×。

### 11.1 Projection-only rollout training

| Candidate | Validation loss | Accepted/proposed | Mean accept | Paired E2E/AR | Parity |
|---|---:|---:|---:|---:|---:|
| S3 parent | 3.770321 | 2/7,650 | 1.003922 | 0.247630× | 16/16 |
| rollout x1, LR1e-5 | 3.436412 | 2/7,650 | 1.003922 | 0.251392× | 16/16 |
| rollout x10, LR3e-6 | 3.106724 | 2/7,650 | 1.003922 | 0.251663× | 16/16 |

x10訓練為106,390 trained-window tokens、135 optimizer steps、258.37 tok/s，peak 4,300 MiB／13.18%。Loss顯著改善但acceptance完全不動，證明projection-only route在此objective下飽和。`LR1e-5` x10另於non-finite gradient停止，未產生checkpoint。

## 12. S6：top-4 draft layers

新增R9700-safe trainer，只解凍layers 24–27與projection：62,923,776 draft params + 20,971,520 projection params；top-4與AdamW皆用FP32 master/state，forward BF16，其他PARD2結構與loss不變。

| Candidate | Validation loss | Accepted/proposed | Paired E2E/AR | Training peak | 結論 |
|---|---:|---:|---:|---:|---|
| top4 x1, LR1e-6 | 3.746498 | 2/7,650 | 0.241585× | 5,264 MiB | 無acceptance改善 |
| top4 x10, LR3e-7 | 3.605319 | 2/7,650 | 0.243531× | 5,264 MiB | 無acceptance改善；不具重現穩定性 |

長top-4訓練有非決定性numeric failure：`LR1e-6`一次non-finite；`LR3e-7`第一次non-finite、第二次完成。Seed不固定，排除單一壞樣本；此checkpoint不得升為穩定候選。

## 13. S11 objective ablation 與 S7 backbone micro

為測試KD改善分布但沒有跨越top-1邊界的假說，新增可選CE/KD權重（預設仍為官方CE0.1/KD1.0），執行projection-only CE1/KD0。x10在LR3e-6及1e-6皆出現non-finite，沒有checkpoint；直接提高CE不是穩定解。

最後執行28-layer backbone micro（embedding/final norm/LM head frozen）：

- Trainable draft layers：440,466,432 params；projection 20,971,520 params。
- FP32 master+Adam估算5.16 GiB；實測training peak 13,264 MiB／40.66%。
- Throughput 225.36 trained tok/s；optimizer checkpoint 5.98 GiB。
- Validation loss 3.770321→3.769338。
- Scorer仍為2/7,650 accepted、mean accept 1.003922、paired E2E/AR 0.244306×、parity 16/16。

## 14. Post-S8最終決策

在zero-overlap benchmark-style target rollouts上，projection x1/x10、top-4 x1/x10、28-layer backbone x1全部維持完全相同的2/7,650 acceptance。Validation CE/KD loss能下降，但沒有轉化為PARD verifier top-1 acceptance；增加trainable capacity或重播epochs不是有效方向。

因此：

- 維持既有PARD2-TI；S3/S5仍只作研究artifact。
- 不重跑S8 formal、不進S9，避免用正式benchmark反覆調參。
- 不再自動擴projection/top-4/backbone epochs。
- 下一個新研究設計必須明確監督acceptance/logit margin或使用更大且多樣的target rollout corpus，並先解決BF16長訓練的非決定性non-finite；這已超出原先凍結的官方PARD2 CE+KD calibration路線。

新增主要artifacts：

```text
qwen3_32b_results/td_calibration/POST_S8_SYSTEMATIC_RESULTS.json
qwen3_32b_results/td_calibration/S10_rollout_alignment/data/manifest.json
qwen3_32b_results/td_calibration/S10_rollout_alignment/data/rollout_sequence_manifest.json
qwen3_32b_results/td_calibration/S10_rollout_alignment/S10_PROJECTION_ONLY_GATE.json
qwen3_32b_results/td_calibration/S6_top4_layers/S10_S6_FINAL_GATE.json
qwen3_32b_results/td_calibration/S6_top4_layers/diagnostic_x10_lr3e7/train_result.json
qwen3_32b_results/td_calibration/S7_backbone/resource_preflight.json
qwen3_32b_results/td_calibration/S7_backbone/rollout_micro_lr1e7/train_result.json
```

MBPP來源：[google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp)。
