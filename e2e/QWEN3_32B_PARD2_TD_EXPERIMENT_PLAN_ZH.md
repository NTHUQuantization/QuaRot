# Qwen3-32B-aligned PARD-2 TD 系統化實驗計畫

> 建立日期：2026-08-28
> 狀態：計畫與gate已凍結；尚未啟動長時GPU訓練
> 機器可讀矩陣：`e2e/qwen3_32b_td_experiment_matrix.json`

## 1. 決策摘要

後續改採逐階段、單一變因、明確go/no-go gate的實驗階梯。正確起點不是直接重訓完整0.8B drafter，而是：

1. 先建立與formal benchmark完全隔離的資料與provenance合約。
2. 用16k tokens驗證32B GPTQ teacher selected-feature extractor的速度、VRAM、磁碟成本與determinism。
3. 依序比較三組32B taps，但只做streaming affine診斷。
4. 僅在出現非零acceptance後，訓練約20.97M參數的projection。
5. projection-only飽和後，才解凍drafter最上方4層；完整drafter訓練是最後選項。
6. checkpoint與threshold完全凍結後，才跑2K→4K→8K smoke與80/80/20 ×3正式測試。

目前已qualified的PARD2-TI保持可部署基線；任何TD實驗失敗都不影響它。

## 2. 凍結基線與研究邊界

下列項目不再隨TD calibration一起變動：

- target固定為`/workspace_root/qwen3_32b_fused_v1_gptq_w4a4kv4_v1`，manifest SHA256為`fd68b66a4142d451b9e633427060d906c6fa1e6b4f2488acc6d41c62e61cf0d0`。
- HIP binary固定為`qwen3_32b_results/kernel/quarot_HIP_gfx1201_g4_m2.so`，SHA256為`8e544408498612b2eae27b6fce9a52e939bded735eba77772692f6aa5fe5cfc5`。
- runtime固定waves 4/2、eager與`QUAROT_FUSED_K1=0`；不重新調kernel dimension。
- drafter架構固定為PARD-2 Qwen3-0.6B/約0.8B、28 layers、hidden 1024。
- target projection固定shape `20480→1024`、scale 0.02；training draft length固定K=16。
- AR與已qualified的8B TI checkpoint作不可變基線。
- 14B-on-32B raw proxy的三資料集`0/480` accepted、mean accept 1.0作失敗基線，不原樣重跑。

一次實驗只能改下列其中一項：

- 32B target taps。
- feature calibration方法。
- 訓練資料量。
- projection是否可訓練。
- 解凍的drafter層數。

target quantization、kernel、K1、taps與trainable scope不得在同一實驗一起變動。本計畫不修改QuaRot或PARD-2 inference架構。

## 3. 既有工具稽核與禁止事項

既有`e2e/pard2_collect_features.py`只能比較「同一模型的fused features與BF16 features」，不是32B-aligned projection trainer：

1. 它同時在一張GPU載入完整BF16 reference與quantized target；32B BF16本身約61.02 GiB，在R9700上不可執行。
2. 它固定使用8B `Pard2Spec()` metadata與taps，無法正確描述14B-on-32B或新的32B-aligned checkpoint。
3. 它的`tune`資料實際取正式HumanEval、GSM8K、MATH-500各前4筆，與formal 80/80/20重疊；`split="tune"`只是一個標籤，沒有真的隔離資料。
4. 它保存兩份raw與兩份projected paired tensors並在RAM重新concat。對32B hidden width，1M tokens的artifact約80.1 GiB，host RAM峰值可能達數百GiB。
5. 它只支援per-channel affine，沒有CE、KD、CAT或projection weight training。

因此該工具不得用於TD32訓練、taps選擇或正式模型選擇。這不改變既有32B AR/TI正式結果；那些checkpoint與formal benchmark沒有使用此collector做32B TD調參。

新collector必須支援：

- 明確的Qwen3-32B profile、target taps與draft/target revisions。
- 真正獨立的data manifest與formal zero-overlap檢查。
- selected-layer hooks，不要求materialize全部65層hidden states。
- token budget、sharding、resume、atomic write與SHA256。
- external AMD-SMI及PyTorch allocator雙重VRAM監測。
- disk/RAM preflight、NaN/Inf與shape/determinism檢查。
- teacher extraction與drafter training分開resident。

## 4. 資料隔離合約

正式benchmark的80/80/20 prompts及其token hashes完全read-only。新資料必須分成互斥的：

- `train`
- `calibration_tune`
- `heldout_validation`
- `formal_benchmark`

每筆資料保存：

- source與immutable revision。
- license。
- sample ID。
- prompt SHA256。
- token-ID SHA256。
- token count。
- split與seed。

任何formal prompt文字或token序列重疊皆為hard stop。formal split不得用來選taps、epoch、loss、threshold或checkpoint。

訓練資料應從PARD-2同類型來源建立分層subset，至少涵蓋code、math與general instruction；實際下載與版本固定在S0完成後才可進GPU。

## 5. 記憶體、磁碟與成本合約

R9700實體VRAM約31.86 GiB；所有GPU stage維持：

- VRAM hard gate：95%。
- hotspot hard gate：110°C。
- memory temperature hard gate：108°C。
- OOM、ROCm fault、NaN、Inf、illegal shape或parity failure立即停止。
- 一次只跑一個GPU工作負載。
- external AMD-SMI peak與PyTorch `max_allocated/max_reserved`都必須通過；1秒external sampling不能取代allocator peak。

目前`/workspace_root`只剩約77 GiB。五組BF16 teacher tensors（四個taps加final hidden）約為47.7 GiB/1M tokens，因此：

- affine screen以streaming sufficient statistics累積`sum(x)`、`sum(y)`、`sum(x²)`、`sum(xy)`，不保存完整paired tensors。
- projection training使用原子、可resume、有SHA256的feature shards。
- 每個shard最多100k tokens、衍生資料最多10 GiB。
- 每次寫入前必須保留至少25 GiB free space。
- 不建立1M-token單一cache，不保存完整151,936維teacher logits。
- derived shard只有在checkpoint、optimizer state與manifest驗證完成後才可回收。

任何預估超過12 GPU-hours的階段，都必須先由上一階段實測重新估價並通過gate。

## 6. 三組target taps候選

14B reference固定使用官方`[-1,-8,-16,-24]`。只改32B taps：

| ID | 32B taps | 假設 |
|---|---|---|
| S2A legacy | `[-1,-8,-16,-24]` | 控制組；測試單純feature affine是否足夠 |
| S2B relative | `[-1,-13,-26,-38]` | 將14B的負索引按40→64 layers相對深度映射 |
| S2C quartile | `[-1,-16,-32,-48]` | 均勻覆蓋64-layer target的low/mid/high features |

三組使用完全相同的paired data、seed、affine方法與heldout evaluator。勝出者只能依獨立heldout validation選擇。

為降低成本，S1一次擷取三組候選的layer union；S2再離線組合各候選。14B reference與32B target分開執行，不可同時resident。

## 7. 逐階段實驗矩陣

| Stage | 單一目的／變因 | 規模 | Go gate | No-go後處理 |
|---|---|---:|---|---|
| S0 Contract | 資料與provenance preflight | CPU only | formal overlap=0；hash/license/revision齊全；磁碟預估通過 | 不進GPU |
| S1 Extractor probe | 只開啟32B selected-feature extraction | 16k tokens | finite、deterministic、每候選width 20480、VRAM/溫度通過 | 修collector，不訓練 |
| S2 Tap/affine screen | 只改32B taps；streamed diagonal affine | 每候選128k tokens | 三個task strata都有accepted token；aggregate mean accept≥2.0；AR parity | 淘汰該taps；不擴資料 |
| S3 Projection micro | 只讓20,972,544參數projection可訓練 | 50k–128k tokens | validation loss下降；各task mean accept≥2；paired E2E不退 | 若兩task仍zero-accept，停止projection-only路線 |
| S4 Projection 500k | 只增加資料量 | 500k tokens | 各task heldout acceptance≥paired TI的90%；E2E不退；單task回退≤2% | 不直接再加資料；分析容量或taps |
| S5 Projection 1M | 只增加資料量 | 1M tokens | 各task acceptance不低於TI；paired TD/TI E2E≥1.03 | acceptance升但E2E不升時先profile overhead |
| S6 Partial unfreeze | 只解凍drafter最上方4層 | 1M tokens | 相對S5 paired E2E再升≥3%；無task回退>2% | 回到projection-only；不進full train |
| S7 Full drafter | 最後才解凍完整約0.8B drafter | 2M–5M tokens | 需另作資源審查 | R9700上不自動啟動 |
| S8 Cache smoke | checkpoint固定，只改2K→4K→8K cache | 1 prompt/task，再16/16/8 heldout | 全數parity；TD/TI E2E≥1.03；95% VRAM gate | 立即停止並回報，不自行offload/降規 |
| S9 Formal | checkpoint與threshold全凍結，只擴evaluation sample | 80/80/20 ×3、8K | parity 540/540；TD/AR CI下界>1；TD/TI E2E≥1.03；CV<5% | 不升為canonical TD |

### S0：Contract與provenance

輸出：

- data manifest與formal exclusion manifest。
- source、dirty tracked files、runtime sources、model與HIP SHA256。
- token、RAM、disk、feature bytes/token預估。
- 每個experiment的唯一ID、parent、唯一變因與artifact root。

S0未完成前，不允許任何GPU calibration。

### S1：16k-token extractor probe

只載入GPTQ 32B teacher，擷取候選layer union與final hidden。記錄：

- selected-feature tokens/s。
- bytes/token與實際shard size。
- external VRAM及Torch allocator peaks。
- 每層feature mean/std/min/max、finite比例。
- 同一input重跑兩次的bitwise或容許誤差determinism。

S1的實測會取代目前4–12 GPU-hour的寬估計，並外推S2/S3成本。

### S2：taps與streaming affine診斷

S2不改projection weight。對同一批input IDs分開抽取14B reference與32B target features，以online sufficient statistics求per-channel affine。每個候選在heldout split上依序評估：

1. raw/projected RMSE與cosine。
2. 三個task strata的acceptance。
3. exact AR parity。
4. paired E2E與steady throughput。

若候選仍有兩個task strata zero-accept，不允許單純靠增加資料量繼續。

### S3–S5：projection-only

由14B `20480→1024` warp初始化，凍結整個0.8B drafter，只訓練：

```text
20,480 × 1,024 + bias = 20,972,544 parameters
```

teacher固定為部署中的GPTQ W4A4KV4 Qwen3-32B。loss維持PARD-2 CAT-weighted CE+KD；為符合32GB限制，允許對vocabulary做數值等價的chunked exact KD，但不得未註明地改成top-k KD。任何loss近似都必須成為獨立實驗，不能與資料量或解凍範圍一起改變。

資料量階梯固定為micro 50k–128k、500k、1M。若4倍資料只帶來小於5%的相對acceptance改善，視為同方法飽和，不繼續盲目加資料。

### S6–S7：解凍drafter

只有S5證明projection capacity不足、且實際speed仍受acceptance限制時，才解凍最上方4層。若S6相對S5沒有至少3% paired E2E收益，不進完整drafter。

S7完整0.8B訓練預估72–240 GPU-hours，且optimizer、activations與full-vocabulary KD可能超過R9700 gate，因此不自動啟動；屆時先回報資源分析，依使用者指引決定本地training plumbing或MI300X級硬體。

### S8–S9：部署與正式驗收

第一個通過S5或S6的checkpoint才可進S8。依序跑2K、4K、8K，先1 prompt/task，再16/16/8 heldout。checkpoint、taps、loss、threshold與runtime完全凍結後才跑S9 formal。

S9成功條件：

- 540/540 output-token exact AR parity。
- 每資料集TD/AR paired speedup bootstrap 95% CI下界大於1。
- paired TD/TI E2E至少1.03。
- run-level speedup CV低於5%。
- VRAM與thermal gates全部通過。

只有S9可將checkpoint升為canonical TD。

## 8. 評估與決策口徑

主要metrics：

- paired end-to-end tokens/s。
- paired steady tokens/s。
- mean accept length。
- draft acceptance與conditional acceptance。
- accepted/proposed tokens。
- exact AR output-token parity。
- external VRAM peak。
- Torch max allocated/reserved。

診斷metrics：

- raw/projected feature RMSE與cosine。
- training/validation CE、KD、CAT losses。
- feature extraction tokens/s。
- cache bytes/token。

最終決策以paired E2E為主，不能只看steady。舊8B affine案例曾改善steady但E2E下降，已證明steady-only gate不可靠。acceptance提高但E2E沒有提高時，先profile projection、draft與target verify overhead，不直接擴大訓練。

## 9. Artifact與命名合約

artifact root固定為：

```text
qwen3_32b_results/td_calibration/
  S0_contract/
  S1_extractor_probe/
  S2A_affine_legacy/
  S2B_affine_relative/
  S2C_affine_quartile/
  S3_projection_micro/
  S4_projection_500k/
  S5_projection_1m/
  S6_top4_layers/
  S8_cache_smoke/
  S9_formal/
```

每個stage至少保存：

- `experiment.json`：ID、parent、hypothesis、唯一變因、所有固定項。
- `data_manifest.json`與zero-overlap結果。
- `source_manifest.json`：git HEAD、dirty source SHA、model/kernel revisions。
- stdout/stderr log。
- AMD-SMI CSV及摘要。
- Torch memory snapshots。
- metrics與gate decision JSON。
- checkpoint/optimizer state SHA256；若該stage有訓練。

失敗artifact不得由retry覆蓋；retry使用`-r1`、`-r2`等新ID。

## 10. 下一個實際動作

下一步只執行S0與S1，不直接開始projection training：

1. 建立真正獨立的data manifests與zero-overlap validator。
2. 建立可選taps、selected hidden hooks、streaming/sharded、fail-close的32B collector。
3. 以16k tokens量測實際tokens/s、bytes/token、external/Torch VRAM與溫度。
4. 根據S1實測更新S2A/S2B/S2C成本。

S1通過後才依序執行S2A、S2B、S2C；不得跳級至1M tokens或完整drafter訓練。
