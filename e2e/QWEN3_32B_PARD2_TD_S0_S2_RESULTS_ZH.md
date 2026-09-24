# Qwen3-32B × PARD2 TD：S0–S2 執行紀錄

日期：2026-08-28（Asia/Taipei）
狀態：S0、S1、S2 已完成；S2 gate 通過；未啟動 S3 或完整 benchmark。
結論：在不更動 QuaRot／PARD2 架構下，PARD2-Qwen3-14B 可作為 Qwen3-32B TD drafter。三組 target-layer mapping 都維持 AR exact parity 並通過 acceptance gate；依預先凍結的 paired held-out 決策規則，`legacy [-1,-8,-16,-24]` 勝出。

## 1. 範圍與硬性 gate

- Target：Qwen3-32B GPTQ W4A4KV4（fused_v1）。
- Drafter：`amd/PARD2-Qwen3-14B`。
- Reference：Qwen3-14B BF16，僅用於離線 feature reference extraction。
- GPU：AMD R9700，總 VRAM 34,208,743,424 bytes，`amd-smi` 顯示 32,624 MiB。
- VRAM gate：`>=95%` 立即中止。
- 熱 gate：hotspot `>=110°C` 或 memory `>=108°C` 立即中止。
- 磁碟 reserve gate：至少保留 25 GiB；完成後尚餘 46,422,269,952 bytes（約 43.23 GiB）。
- 所有正式 workload 均為 batch 1；calibration/reference 最大序列長度 512；held-out cache length 2,048。
- 本階段只做資料契約、feature extraction/calibration 與獨立 held-out scorer；沒有訓練 PARD2，也沒有啟動 S3 或三個完整 benchmark。

環境固定值：

```text
QUAROT_FUSED_K1=0
QUAROT_QWEN3_32B_GROUPED_NWAVES=4
QUAROT_QWEN3_32B_MULTI_NWAVES=2
HIP kernel: quarot_HIP_gfx1201_g4_m2.so
HIP SHA256: 8e544408498612b2eae27b6fce9a52e939bded735eba77772692f6aa5fe5cfc5
git HEAD: 8d8c85a07238a34c3583880e717a08523b0f040f
git branch: fused_pard2
S0 worktree state: dirty, 42 porcelain entries
```

## 2. S0：資料與實驗契約

### 2.1 Calibration tune set

固定為 128,400 tokens、870 samples：

| Stratum | Dataset revision | Samples | Tokens |
|---|---|---:|---:|
| general | `HuggingFaceH4/ultrachat_200k@8049631…` / `train_sft` | 234 | 42,686 |
| code | `ise-uiuc/Magicoder-Evol-Instruct-110K@b0079…` / `train` | 268 | 42,939 |
| math | `open-r1/OpenR1-Math-220k@e4e141…` / `train` | 368 | 42,775 |
| **合計** |  | **870** | **128,400** |

- Tune JSONL SHA256：`c49be784a0c27bdbc86cd991ea11330ec1069af6409c47f0fb4e57d9dff70ded`
- Data manifest SHA256：`d51b70cdca319190f9d78f295636d649b9cace66b790fe58887aa1228e4ee504`

### 2.2 Independent held-out set

- 12 prompts、1,779 input tokens；general/code/math 各 4 prompts。
- Held-out JSONL SHA256：`acc5fbe874c776d274f1e10eb833d0b07a06f84c6090e14fa56bdcf3f8640b15`
- Tune/held-out exact normalized-text overlap：0。
- Token-ID hash overlap：0。
- 長度至少 80 characters 的 formal substring overlap：0。
- Gate：**PASS**。

### 2.3 Frozen artifacts

| Artifact | Revision / SHA256 |
|---|---|
| Qwen3-32B GPTQ target index | `3e9f91f595a336ed51bb287dcaa4264fae2f223d73ef3f43c4d47ae946c834fa` |
| PARD2-Qwen3-14B draft revision | `679eff0b65ffaf5abd2dadd21a17909562935798` |
| Draft warp | `24856868496b96df7ca3176235c488308990f7d7c38605f8e0b5e2361ee529b0` |
| Qwen3-14B BF16 reference revision | `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Reference index | `62d7ad35757bae5e7baa452cb1483178b7daa50e869e923226b8da10871f7ebc` |
| S0 contract | `101f1942b88bf885e590825293f336b0809b2ac77d9b3dbad09e8bcb8853f2a0` |

S0 gate：**PASS**。

## 3. S1：32B selected-feature extractor probe

### 3.1 初次失敗與根因

原始 S1 在 selected feature 上出現 NaN/Inf，但進一步逐層診斷證明：

- Target `model` output 與完整 logits 全部 finite。
- 所有被 hook 的 raw layer hidden states 全部 finite。
- 最終 `-1` feature 可正常恢復。
- 其餘 union taps 在原始 FP16 inverse-Hadamard 恢復時，各出現 1 個 `+Inf / 680,960 elements`。
- raw finite max 約 442；未縮放恢復的 finite max 可到約 2,528，因此是 FP16 inverse-Hadamard 中間加總 overflow，不是模型、GPTQ 權重、資料或 VRAM 錯誤。

### 3.2 數值修正

離線 collector 與 held-out scorer 使用同一個安全恢復路徑：

1. inverse-Hadamard 前以 power-of-two `16` 預縮放。
2. 仍執行原 FP16 inverse-Hadamard kernel。
3. 輸出轉 FP32 後乘回 `16` 並套用 rotation signs。
4. 最終 RMSNorm 以 FP32 計算。
5. 每次抽取都強制 finite check。

這是數值工具修正，不改 target/drafter、taps、projection、warp、QuaRot 或 PARD2 架構。合成測試相對 FP32 reference：BF16 inverse RMSE 1.668，scaled-FP16 inverse RMSE 0.214（signal RMS 約 449.96），因此採 scaled FP16。

修正 source hashes 與驗證結果另存於 `numeric_correction_manifest.json`。

### 3.3 S1 正式結果

測試 union taps：`[-1,-8,-16,-24,-13,-26,-38,-32,-48]`。

| Metric | Result |
|---|---:|
| Processed tokens | 16,000 |
| Extractor throughput | 179.681 tok/s |
| Determinism | 4/4 bit-exact；max abs 0；RMSE 0 |
| 所有 taps finite | Yes |
| External peak VRAM | 27,521 MiB / 84.36% |
| Peak hotspot / memory | 74°C / 55°C |
| Runtime elapsed（不含前置載入） | 89.047 s |

S1 artifact SHA256：`b8fc0f7d811249df1fa695a58488bf6809aef8527cc7a98872e16423d45f1061`。
S1 gate：**PASS**。

## 4. S2-A：14B reference extraction

- Reference taps：`[-1,-8,-16,-24]`。
- 128,000 tokens，切成 8 個 16K-token shards。
- 總 feature shard bytes：5,243,067,176。
- 8/8 shard SHA256 重算吻合。
- Throughput：296.012 tok/s；抽取 elapsed 432.415 s。
- External peak VRAM：28,845 MiB / 88.42%。
- Peak hotspot / memory：97°C / 70°C。
- Manifest SHA256：`e40b1911ebb1aaea5f700b4f6378b0b6dbaa735c1ce6ad1543825792b088391d`。

Reference extraction gate：**PASS**。這證明 Qwen3-14B BF16 可在 32GB R9700 上以 batch 1、sequence 512 執行離線 reference extraction，但僅剩約 3.7 GiB 外部 VRAM 裕量，不建議提高 batch 或同卡併載 32B target。

## 5. S2-B：128K affine fitting

Reference feature 固定為 14B 的 `[-1,-8,-16,-24]`；比較三組 32B target taps：

| Candidate | 32B target taps | RMSE before | RMSE after | Cosine before | Cosine after | Zero-var channels |
|---|---|---:|---:|---:|---:|---:|
| legacy | `[-1,-8,-16,-24]` | 15.0696 | 9.9378 | 0.82724 | 0.83814 | 0 |
| relative | `[-1,-13,-26,-38]` | 14.5598 | **9.9234** | 0.82077 | **0.83866** | 0 |
| quartile | `[-1,-16,-32,-48]` | 14.5624 | 9.9540 | 0.81744 | 0.83756 | 0 |

- Fitting throughput：241.270 tok/s；elapsed 530.525 s。
- External peak VRAM：27,521 MiB / 84.36%。
- Peak hotspot / memory：98°C / 78°C。
- Fit manifest SHA256：`f19306d0fa66de7a5c47d22eb22497c093fd48eb19d98f1333518b146d725796`。

離線 fit 由 relative 微幅領先，但三者差距太小，依契約不得用 RMSE 單獨選 winner。

## 6. S2-C：independent held-out scorer

設定：12 prompts（每 stratum 4）、每 prompt 固定生成 32 tokens、cache 2,048。AR 只執行一次；每個 TD 候選分開載入與監測。Gate 條件：

1. TD output IDs 必須與 AR 完全一致。
2. general/code/math 每一層都必須有 accepted draft tokens。
3. aggregate mean accept length 必須 `>=2.0`。

### 6.1 AR baseline

| Metric | AR |
|---|---:|
| 12 prompts exact completion | 12/12 |
| Median steady | 16.325 tok/s |
| Median E2E | 15.051 tok/s |
| External peak VRAM | 27,212 MiB / 83.41% |
| Peak hotspot / memory | 82°C / 63°C |

### 6.2 TD candidate results

| Candidate | Exact AR parity | Mean accept | General / Code / Math accepted | Paired median steady | Paired median E2E | Peak VRAM | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **legacy** | 12/12 | **4.874** | 89 / 105 / 143 | **1.0887×** | **1.0017×** | 28,603 MiB / 87.67% | PASS |
| relative | 12/12 | 4.495 | 87 / 109 / 136 | 0.9877× | 0.9118× | 28,603 MiB / 87.67% | PASS |
| quartile | 12/12 | 4.047 | 85 / 99 / 142 | 0.8574× | 0.7790× | 28,603 MiB / 87.67% | PASS |

分層 mean accept length：

| Candidate | General | Code | Math |
|---|---:|---:|---:|
| legacy | 3.070 | 4.750 | 9.938 |
| relative | 2.891 | 4.633 | 8.158 |
| quartile | 2.771 | 3.676 | 7.455 |

Decision rule 是「先通過 gate，再最大化 paired median E2E，最後以 mean acceptance tie-break」。因此 S2 winner：**legacy `[-1,-8,-16,-24]`**。

注意：legacy 的各組未配對 raw median E2E 為 13.756 tok/s，AR raw median 為 15.051 tok/s；但逐 prompt ratio 的 median 是 1.0017×。這兩種統計不等價。樣本只有 12 個且每題只生成 32 tokens，1.0017× 不應解讀為已確認的 benchmark speedup，只代表它依預先凍結的 S2 paired scorer 勝出並有資格進 S3。

quartile 初跑的 workload 結果有效，但外部 monitor 在第 54 秒提前退出，只覆蓋 CPU 載入期，故不採其 VRAM 證據。以新 prefix 完整重跑後：

- output IDs 與初跑完全一致。
- per-step acceptance lengths 與初跑完全一致。
- 完整 monitor 219 samples，峰值 28,603 MiB / 87.67%。
- retry peak hotspot / memory：74°C / 56°C。

S2 decision artifact SHA256：`cfbc178f89d49857f3eab96b27da13b633b456e05f0755ea54fbb5def635f03e`。
S2 gate：**PASS**。

## 7. 判讀與後續邊界

1. **14B drafter 是目前合理選擇。** 它提供與 32B 相同 hidden size 5,120 的 feature contract；S2 中三組 mapping 全部 exact parity、各 stratum 均有 acceptance，已排除「14B drafter 完全不適用 32B」的假設。
2. **相同負索引比按深度比例重映射更好。** 32B 雖有 64 layers，legacy `[-1,-8,-16,-24]` 在 held-out acceptance 與 paired E2E 都優於 relative/quartile。單純因 layer count 不同而把 taps 拉深，反而降低 drafter 可用性。
3. **離線 feature RMSE 不能代理最終速度。** relative 的 affine RMSE/cosine 略好，held-out acceptance/E2E 卻輸給 legacy；後續 mapping 或 calibration 必須以 exact-parity TD scorer 收尾。
4. **general 是最弱 stratum，math 最強。** legacy mean accept 分別 3.07 與 9.94，呼應先前 GSM8K 類任務退化較明顯的觀察；下一階段應優先擴充 general／短推理／格式敏感題，而不是再提高 math 比例。
5. **目前瓶頸不是 VRAM。** TD held-out 峰值 87.67%，約保留 2.39 GiB 到 95% gate；仍不適合顯著提高 batch/cache，但 S3 的 batch-1 benchmark 可在相同設定下進行。
6. **目前尚不能宣稱 32B TD speedup。** legacy 只在小型 paired scorer 上略過 1.0×；需 S3 完整 benchmark、較長 generation 與 kernel breakdown 才能判斷是否真正改善先前 GSM8K/HumanEval/MATH-500 的退化。

建議的下一個明確步驟（需另行指示後才執行）：固定 legacy taps 與其 calibration，進 S3 小規模 benchmark gate；先以 general/GSM8K 類 workload 驗證，再依序 code、math。若 S3 E2E 仍退化，優先量測 feature restore、warp、verify-step 與 drafter launch overhead，不先更動 PARD2 架構。

## 8. 主要 artifact 索引

```text
qwen3_32b_results/td_calibration/S0_contract/contract.json
qwen3_32b_results/td_calibration/S0_contract/calibration_tune.jsonl
qwen3_32b_results/td_calibration/S0_contract/heldout_validation.jsonl
qwen3_32b_results/td_calibration/S1_extractor_probe/nan_diagnostic.json
qwen3_32b_results/td_calibration/S1_extractor_probe/probe_safe_r1.json
qwen3_32b_results/td_calibration/S2_reference/reference_manifest.json
qwen3_32b_results/td_calibration/S2_affine/fit_manifest.json
qwen3_32b_results/td_calibration/S2_affine/{legacy,relative,quartile}/calibration.pt
qwen3_32b_results/td_calibration/S2_heldout/ar.json
qwen3_32b_results/td_calibration/S2_heldout/td_{legacy,relative}.json
qwen3_32b_results/td_calibration/S2_heldout/td_quartile_r1.json
qwen3_32b_results/td_calibration/S2_heldout/decision.json
qwen3_32b_results/td_calibration/numeric_correction_manifest.json
```

測試：`tests/test_qwen3_32b_td_numeric.py` + `tests/test_qwen3_32b_td_stages.py`，共 4/4 passed。
