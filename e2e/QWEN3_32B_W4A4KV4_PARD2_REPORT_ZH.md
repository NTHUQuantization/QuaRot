# Qwen3-32B QuaRot W4A4KV4 × fused_v1 × PARD2 實驗紀錄

> 最終狀態（2026-08-28）：streaming GPTQ W4A4KV4、checkpoint audit、BF16/RTN/GPTQ 分布比較、2K/4K/8K smoke、8K AR/PARD2-TI 80/80/20 × 3 sweeps、95% VRAM gate、獨立 task scoring，以及 PARD2-Qwen3-14B-on-Qwen3-32B experimental TD proxy smoke均已完成。本文是本次工作的唯一主報告；RTN僅保留為基線。

## 1. 結論摘要

- Qwen3-32B GPTQ W4A4KV4 checkpoint 已建立於 `/workspace_root/qwen3_32b_fused_v1_gptq_w4a4kv4_v1`。量化採逐層 streaming，沒有在 32 GB R9700 上載入完整 BF16 模型。
- GPTQ 64/64 layers 完成，耗時 4:16:59；量化 VRAM 峰值 15,288 MiB（46.86%），沒有超過 90% gate。
- 嚴格 audit 通過：65 shards、1026 keys、18,732,843,008 B tensor payload；GPTQ manifest 為 `fd68b66a4142d451b9e633427060d906c6fa1e6b4f2488acc6d41c62e61cf0d0`。
- audit 發現量化時 q/k norm 被 calibration layer 轉成 FP16。已用 immutable BF16 source 原子修復 64 shards、128 tensors，且逐 tensor exact match；producer 也已修正，避免重建時再發生。
- 三個固定 prompt 的 aggregate 分布診斷支持「GPTQ 比 clipped RTN 更接近 BF16」：RMSE、MAE、KL、TV 全部下降，cosine 與 top-1/top-5 提升；但個別 prompt 並非每個指標都改善，因此不能把三 prompt 診斷當作完整 accuracy 結論。
- GPTQ HumanEval、GSM8K、MATH-500 的 2K-cache AR/TI smoke 均 32/32 output-token exact match；另完成 HumanEval 2K/4K/8K cache scaling smoke，三組也全為 exact parity。
- cache scaling 最大外部 VRAM 為 8K TI 的 28,619 MiB（87.72%）；低於使用者指定的 8K 95% gate，也低於原 90% gate。沒有 OOM、ROCm fault、NaN 或非法 shape。
- GPTQ 不改 tensor shape。Qwen3-32B 現有 exact-shape dispatch、grouped/multi waves 4/2 與最終 HIP binary 仍適用；本階段沒有證據支持再次調整 kernel dimension。維持 `QUAROT_FUSED_K1=0`，因既有 K1-on 路徑曾造成 AR/TI token 分岔。
- 正式 8K 測試共完成 AR 540 + TI 540 = 1,080 runs；TI/AR paired median steady speedup 為 HumanEval 1.5185×、GSM8K 1.4448×、MATH-500 1.4505×，三者 bootstrap 95% CI 下界均大於 1，且全數 token exact parity。
- 正式最高外部 VRAM 為 TI 28,619 MiB（87.72%），距 95% gate 7.28 percentage points；無 OOM、ROCm fault、VRAM/thermal safety event、NaN 或非法 shape。
- 獨立 scorer 結果：HumanEval execution pass@1 50.00%、GSM8K numeric exact 58.75%、MATH-500 conservative normalized exact 10.00%。MATH 分數受 256-token 截斷與保守等價判定影響，應視為可重現下界而非寬鬆語義分數。
- Qwen3-14B-aligned TD warp在32B上shape/preflight、AR parity與VRAM均通過，但三資料集各1個2K/32-token smoke皆為0/480 accepted draft tokens、mean accept 1.0，steady僅為AR的22.6–23.1%；依gate取消8K長測，判定不可作canonical TD。

## 2. 範圍、硬體界線與 PARD2 32B 對照

本階段維持 QuaRot randomized rotation、W4/A4/KV4 定義與 PARD2 speculative data flow。允許 Qwen3-32B 固定維度的 GEMM dispatch、tile、group、wave 與 fused width specialization；不自動更換 draft、改 speculative 演算法、offload、縮 batch/context 或放寬硬體 gate。

R9700 實體 VRAM 為 34,208,743,424 B（31.86 GiB）；一般 90% hard ceiling 為 30,787,869,082 B，約 29,361.6 MiB。2026-08-27 使用者僅針對本次 8K smoke 將 gate 改為 95%（約 30,992.8 MiB）；OOM、ROCm fault、checkpoint/parity 錯誤、NaN、非法 shape 或超過當次 gate 仍須停止並回報。

PARD2 論文 Table 2 確實包含 Qwen3-32B，但為 target-independent 模式。論文在 temperature 0、thinking disabled、PARD-2 K=16 下，HumanEval/GSM8K/MATH-500/MT-Bench 平均 speedup 4.68×、平均 accept length 5.75；環境是 vLLM、TP=2、兩張 A100-40GB。這與本次單張 R9700、W4 target、官方 8B TI draft、32-token smoke 不同，不能直接作驗收線。官方釋出也沒有 Qwen3-32B-aligned TD projection，因此本輪只執行 canonical AR/TI，不執行 14B cross-target TD proxy。

- Paper: https://arxiv.org/pdf/2605.08632
- Official repository: https://github.com/AMD-AGI/PARD

## 3. 固定環境與模型 provenance

| 項目 | 值 |
|---|---|
| Git branch / HEAD | `fused_pard2` / `8d8c85a07238a34c3583880e717a08523b0f040f`；工作樹含本次未提交修改 |
| 容器 | `qwen3_32b_quarot_clean`；以 UID 3016 / GID 3000 執行 benchmark |
| GPU | AMD R9700，gfx1201，31.86 GiB VRAM |
| PyTorch / HIP | `2.9.1+git5bc97ba` / `7.2.26015-fc0010cf6a` |
| Transformers / AMD SMI | `4.57.6` / `26.2.1+fc0010cf6a` |
| Qwen source | `Qwen/Qwen3-32B` revision `9216db5781bf21249d130ec9da846c4624c16137` |
| source dtype / size | BF16；index total size 65,524,246,528 B（61.02 GiB）；707 tensors / 17 shards |
| source config SHA256 | `97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb` |
| source index SHA256 | `bed42c6c55274bc08a1f616bceb3bcb84b3f02cb6584c573bd18c6519291ecd0` |
| TI draft | `amd/PARD2-Qwen3-8B` revision `67a1516c8f6fc145cda99916799a0cbb3a4af135` |

Qwen3-32B 架構：hidden 5120、FFN 25600、64 layers、64 Q heads、8 KV heads、head dim 128、vocab 151936、max positions 40960，embedding 與 LM head 不共享。

## 4. Streaming GPTQ W4A4KV4

### 4.1 固定量化契約

| 項目 | 設定 |
|---|---|
| method | streaming GPTQ v5；逐 global/layer shard，原子寫入、可 resume |
| calibration | WikiText-2 train，seed 0，128 samples × 2048 tokens |
| calibration token SHA256 | `ec0a043b2c48080ebb5cce82773928c13562007527fc443b1fd8f651a02820d8` |
| W4 | 4-bit、per-output-channel symmetric、groupsize -1、MSE clipping |
| GPTQ | percdamp 0.01、act-order false、static-groups false |
| rotation | ROCm GPU、FP32、seed 0；HadK remainder 40、inner 128 |
| A4 | runtime per-row symmetric，clip ratio 0.9 |
| KV4 | runtime per-token/per-head asymmetric，native GQA |
| checkpoint | format v2、`grouped_h256_v1` |

實際命令的核心參數：

```bash
python e2e/checkpoint_utils/quantize_checkpoint.py \
  --model /workspace_root/.hf_cache/pard/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
  --output /workspace_root/qwen3_32b_fused_v1_gptq_w4a4kv4_v1 \
  --quant-method gptq --dataset wikitext2 --nsamples 128 --seqlen 2048 \
  --seed 0 --w-groupsize -1 --w-clip --percdamp 0.01 \
  --rotation-device cuda --rotation-dtype float32
```

逐層設計的主要記憶體來源是 calibration activation in/out（各約 2.50 GiB）與最大 down-proj Hessian（25600² FP32，約 2.44 GiB）；完成一層即釋放並寫 shard，因此不需要同時 resident 61.02 GiB BF16 source。

### 4.2 量化結果與硬體監測

- 64/64 layers、程序 exit 0、耗時 4:16:59；沒有殘留 partial/temp shard。
- checkpoint file bytes 18,749,080,706；65 個 safetensors file bytes 18,732,986,344；tensor payload 18,732,843,008。
- 量化 window 14,509 samples，估計 coverage 93.98%；VRAM peak 15,288 MiB（46.86%），沒有任何 sample 超過 90%。
- hotspot peak 108°C、memory temperature peak 92°C；沒有 sample 達到設定的 hotspot 110°C 或 memory 108°C 停止線。
- AMD SMI CSV 的 `power_usage` 出現 558 W 等超過裝置 `max_power=300 W` 的異常值（149 rows >330 W），與欄位上限不相容；因此 power 原始欄位保留供稽核，但不把 558 W 解讀為可信實體功耗，也不據此宣稱硬體故障。

原始監測與摘要位於 `qwen3_32b_results/quantization/gptq/amd_smi.csv`、`amd_smi_summary.json`。

### 4.3 嚴格 artifact audit 與 norm 修復

最終 audit 結果：65 shards、1026 weight-map keys、所有 header shape/dtype/index mapping/signature/source provenance 均通過；config SHA256 為 `b310f60888f0b66b977b47268fe12dacaa1cd530722ab03cccf31bb86d43e36d`，index SHA256 為 `3e9f91f595a336ed51bb287dcaa4264fae2f223d73ef3f43c4d47ae946c834fa`，manifest SHA256 為 `fd68b66a4142d451b9e633427060d906c6fa1e6b4f2488acc6d41c62e61cf0d0`。

第一次 audit 發現 64 層 q_norm/k_norm 共 128 tensors 是 FP16，而 immutable source 是 BF16。根因是 GPTQ calibration model layer 將 norms 轉成 FP16，converter 又直接保存該 dtype。處理如下：

1. 先確認 FP16 tensors 轉回 BF16 後與 immutable source bit-exact。
2. 逐 shard 以 source BF16 norm 原子替換，64 shards / 128 tensors 全部完成。
3. 重跑 strict audit，`exact_source_match=true`。
4. 修改 streaming GPTQ producer，使未來直接保存 source BF16 q/k norms；新增 `repair_qwen3_32b_gptq_norms.py` 作可重現 migration。

RTN 基線也在 GPTQ 修改後重跑 regression audit，manifest 仍為 `0f280fba6ada3fb2fd9d0abeee67b914bd38c43f7c0ee323e41891418b099eed`。

## 5. BF16 / RTN / GPTQ 輸出分布

比較使用 HumanEval offset 0（107 tokens）、GSM8K offset 0（62 tokens）、MATH-500 offset 0（70 tokens），各取最後位置 151,936 維 logits。BF16 以逐層 streaming reference 執行，不載入完整模型；RTN/GPTQ 使用相同最終 HIP binary。三份 logits 均 finite。

| aggregate（3 prompts 平均） | RTN | GPTQ | GPTQ 相對判定 |
|---|---:|---:|---|
| logit RMSE ↓ | 2.321910 | 1.430828 | 較低 |
| logit MAE ↓ | 1.960624 | 1.153819 | 較低 |
| cosine ↑ | 0.936062 | 0.969442 | 較高 |
| KL(BF16‖candidate) ↓ | 0.343273 | 0.222838 | 較低 |
| total variation ↓ | 0.306843 | 0.214582 | 較低 |
| top-5 overlap ↑ | 4.0000 | 4.6667 | 較高 |
| top-1 match | 2/3 | 3/3 | 較高 |

| dataset | method | RMSE | MAE | cosine | KL | TV | top-1 | top-5 overlap |
|---|---|---:|---:|---:|---:|---:|---|---:|
| HumanEval | RTN | 3.2473 | 2.9417 | 0.9458 | 0.5296 | 0.3646 | match | 5 |
| HumanEval | GPTQ | 0.8304 | 0.6575 | 0.9935 | 0.3248 | 0.3028 | match | 5 |
| GSM8K | RTN | 2.1679 | 1.7189 | 0.8902 | 0.1970 | 0.1836 | match | 3 |
| GSM8K | GPTQ | 1.4871 | 1.1773 | 0.9490 | 0.3322 | 0.2877 | match | 4 |
| MATH-500 | RTN | 1.5505 | 1.2213 | 0.9721 | 0.3032 | 0.3724 | mismatch | 4 |
| MATH-500 | GPTQ | 1.9749 | 1.6267 | 0.9658 | 0.0115 | 0.0532 | match | 5 |

因此使用者提出的方向在 aggregate 診斷上成立，但需保留兩個 caveat：GPTQ 的 GSM8K KL/TV 比 RTN 高，MATH-500 RMSE/MAE 也比 RTN 高；三 prompt 不足以代表完整 task accuracy。

artifact SHA256：BF16 reference `f5b9fb58224dfec73426a08b54c1d23d9dc3e6dc7d7301df926def93fbc986cf`；RTN target `d40eeee28235232cfdeb65fcab05ae4c7716939c841c98065b7d7b07ac4537b7`；GPTQ target `a9ff6c0e20b4a463440af68106ce2c6882801dc9c4427539e52c14492e425d36`。

外部 VRAM peaks：BF16 streaming 1,744 MiB（5.35%）、RTN 27,058 MiB（82.94%）、GPTQ 27,057 MiB（82.94%）。第一次 BF16 diagnostic 因 decoder helper 錯把 Tensor 當 tuple 而失敗，修正後 retry2 成功；失敗 log/CSV 均保留。

## 6. Kernel / dimension 稽核與設定

GPTQ 只改 W4 codes，不改 weight shape、scale shape、A4/KV4 contract 或 PARD2 data flow，因此 RTN 階段完成的 Qwen3-32B specialization 可直接沿用。

| 路徑 | 固定維度 | 最終設定／判定 |
|---|---:|---|
| hidden QuaRot | 5120 | generalized HadK：K=40、inner H128；random-sign roundtrip 通過 |
| Q projection | 5120→8192 | exact N8192 B-prepack specialization |
| K/V projection | 5120→1024 | exact N1024 specialization、native GQA |
| O/down output | N5120 | Qwen3-32B exact B-prepack specialization |
| gate/up output | N25600 | Qwen3-32B exact B-prepack specialization |
| attention | 64 Q heads × 128 | fused width 8192 boundary 已測 |
| FFN transform | 25600 = 100×256 | `grouped_h256_v1`，無 padding |
| grouped waves | — | `QUAROT_QWEN3_32B_GROUPED_NWAVES=4` |
| multi waves | — | `QUAROT_QWEN3_32B_MULTI_NWAVES=2` |
| fused K1 | — | `QUAROT_FUSED_K1=0`，維持 AR/TI numerical contract |

最終 gfx1201 HIP binary：`qwen3_32b_results/kernel/quarot_HIP_gfx1201_g4_m2.so`，SHA256 `8e544408498612b2eae27b6fce9a52e939bded735eba77772692f6aa5fe5cfc5`；kernel correctness 178/178 passed。既有 wave sweep 顯示 grouped g4 優於 g2，multi m2 對三個核心 case 合計約改善 0.42%。GPTQ smoke 的實際 extension path/SHA 與 wave flags均由 benchmark fail-close 記錄。

本階段結論是「不需要因 GPTQ 再調 dimension」。下一輪優先事項應是擴大 accuracy/acceptance sample，而不是再改 5120/8192/1024/25600 dispatch。K1-on 若要恢復，應獨立修復 fused RoPE+KV4 numerical contract，不能與 GPTQ accuracy 結論混在一起。

## 7. GPTQ 三資料集 AR/TI smoke

固定條件：batch 1、greedy、正常 EOS、32 generated tokens、cache 2048、1 warmup、1 sweep、`limit=1`、eager、K1-off、waves 4/2。TI 使用 pinned PARD2-Qwen3-8B draft。所有結果 `formal_protocol=false`、`qualified=false`。

| dataset | AR steady tok/s | TI steady tok/s | TI/AR | AR E2E tok/s | TI E2E tok/s | TI/AR | token parity |
|---|---:|---:|---:|---:|---:|---:|---|
| HumanEval | 15.8252 | 10.3327 | 0.6529× | 10.9112 | 9.1466 | 0.8383× | 32/32 exact |
| GSM8K | 15.7974 | 13.0226 | 0.8244× | 10.8713 | 12.3843 | 1.1392× | 32/32 exact |
| MATH-500 | 15.4991 | 16.2630 | 1.0493× | 10.5702 | 13.3412 | 1.2622× | 32/32 exact |

| dataset | mean accept length | draft acceptance | conditional acceptance | target forwards | draft forwards |
|---|---:|---:|---:|---:|---:|
| HumanEval | 3.2500 | 0.1500 | 0.6923 | 13 | 13 |
| GSM8K | 5.2222 | 0.2815 | 0.8261 | 10 | 10 |
| MATH-500 | 4.0000 | 0.2000 | 0.7500 | 9 | 9 |

此結果顯示 TI 效益主要受 prompt-specific acceptance 影響：HumanEval 此 prompt 變慢，GSM8K steady 變慢但 E2E 變快，MATH-500 有小幅 steady 與明顯 E2E gain。不能用單 prompt 對 PARD2 或 GPTQ 作總體速度排名。

### 7.1 VRAM、溫度與安全界線

| dataset | AR peak MiB | AR % | TI peak MiB | TI % | 最高 hotspot / memory |
|---|---:|---:|---:|---:|---|
| HumanEval | 27,065 | 82.96% | 28,201 | 86.44% | 69°C / 48°C |
| GSM8K | 27,064 | 82.96% | 28,201 | 86.44% | 60°C / 48°C |
| MATH-500 | 27,064 | 82.96% | 28,201 | 86.44% | 63°C / 50°C |

這六個 2K run 均低於 90% hard ceiling。最壞 TI 距 gate 約 1,160 MiB；這是可執行但不寬裕的 headroom，所以後續仍應一次只跑一個模型工作負載。監測 CSV 的最高 power raw value 為 HumanEval TI 290 W；benchmark 溫度遠低於量化階段。

8192 cache 的早期 resident-size 加總曾保守投影超過 90%，所以三資料集首輪採 2048；後續實際 cache scaling smoke 證明該加總高估 concurrent peak，詳見第 10 節。三資料集完整 tokenization audit 的最大需求（input + 256 output + draft-k 15）分別為 HumanEval 690、GSM8K 418、MATH-500 633 tokens。

### 7.2 可重現執行要點與失敗紀錄

穩定執行必須：以 UID/GID 3016:3000 跑、從 `/tmp` 使用 `python -m e2e.benchmark_pard2`、把 final extension overlay 放在 `PYTHONPATH` 最前、使用獨立可寫 Triton cache，並傳入 expected HIP SHA。

```bash
TRITON_CACHE_DIR=/tmp/qwen3-32b-uid3016-triton-cache \
QUAROT_FUSED_K1=0 \
QUAROT_QWEN3_32B_GROUPED_NWAVES=4 \
QUAROT_QWEN3_32B_MULTI_NWAVES=2 \
PYTHONPATH=/tmp/qwen3-32b-ti-runtime-option1:/workspace_root/fused_v1:/workspace_root/fused_v1/third-party/hadacore \
python -m e2e.benchmark_pard2 \
  --mode ar \
  --dataset math_500 \
  --target /workspace_root/qwen3_32b_fused_v1_gptq_w4a4kv4_v1 \
  --tokenizer /workspace_root/.hf_cache/pard/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137 \
  --generated-tokens 32 --max-cache-len 2048 --warmups 1 --sweeps 1 --limit 1 \
  --compile-mode eager --benchmark-profile qwen3_32b \
  --expected-hip-sha256 8e544408498612b2eae27b6fce9a52e939bded735eba77772692f6aa5fe5cfc5 \
  --output <output.json>
```

TI 將 `--mode` 改成 `pard2-ti`，並加入上述 pinned 8B draft 路徑。

已保存的 preflight/啟動失敗：HumanEval AR 曾因 root 執行觸發 Git dubious ownership、direct script import 載入錯誤 repo HIP binary、root Triton cache 權限失敗；MATH-500 AR 曾將 mode 寫成 `autoregressive`，CLI 在模型載入前拒絕。這些失敗均未納入性能數字，logs/CSVs 保留，修正後成功 run 不覆蓋其證據。

## 8. 程式與測試變更

- `e2e/checkpoint_utils/streaming_gptq.py`：streaming GPTQ、source BF16 q/k norm preservation、method metadata。
- `e2e/repair_qwen3_32b_gptq_norms.py`：既有 GPTQ checkpoint 的原子 BF16 norm migration。
- `e2e/audit_qwen3_32b_checkpoint.py`：method-aware strict GPTQ/RTN contract、shape/dtype/index/manifest audit。
- `e2e/benchmark_pard2.py`：strict GPTQ v5 preflight、expected HIP SHA、runtime/source/token provenance、cache length profile。
- `e2e/compare_qwen3_32b_output_distribution.py`：BF16 streaming reference 與 RTN/GPTQ distribution comparator。
- `e2e/attach_amd_smi_monitor.py`、`e2e/run_qwen3_32b_formal_8k.sh`：外部監測回填、95% fail-close 與六組 sequential formal runner。
- `e2e/run_qwen3_32b_proxy_experiment.sh`：AR/TI/TD experimental matched runner；TD固定pinned 14B snapshot、cross-target metadata與相同95%/thermal fail-close。
- `e2e/score_qwen3_32b_tasks.py`、`e2e/run_humaneval_sandbox.py`：ground-truth 對齊、GSM8K/MATH scorer 與隔離 HumanEval execution scorer。
- `tests/test_streaming_gptq.py`、`tests/test_pard2_benchmark_contract.py`、`tests/test_qwen3_32b_output_distribution.py`：新增/更新回歸測試。
- `tests/test_amd_smi_monitor.py`、`tests/test_qwen3_32b_task_scorer.py`：監測器與 scorer regression tests。

最終重跑 streaming GPTQ、benchmark contract、distribution tests 共 43/43 passed；正式結果完成後再重跑 AMD-SMI monitor、task scorer、benchmark contract 共 34/34 passed；先前 distribution comparator focused tests 4/4 passed；最終 HIP kernel suite 178/178 passed。34-test 首次指令因 `PYTHONPATH` 未含既有 hadacore 而在 collection 前停止（0 tests executed），補上與 benchmark 相同的 hadacore/runtime overlay 後全數通過。pytest 只有 Triton deprecation 與無法寫入既有 root-owned `.pytest_cache` 的 warning，不影響測試結果。

## 9. Raw artifacts 索引

根目錄：`/workspace_root/fused_v1/qwen3_32b_results/`

- GPTQ 量化：`quantization/gptq/quantize.log`、`amd_smi.csv`、`amd_smi_summary.json`、`artifact_audit.json`、`norm_repair.json`
- RTN regression：`quantization/rtn/artifact_audit_regression_after_gptq.json`
- 分布比較：`distribution/{bf16_reference,rtn_target,gptq_target}.pt`、對應 logs/CSVs、`comparison.json`、`comparison.log`
- GPTQ smoke：`benchmark/gptq_smoke/{ar,pard2_ti}_{humaneval,gsm8k,math_500}.json`、logs、AMD SMI CSV；HumanEval AR 成功監測檔為 `ar_humaneval_retry4_amd_smi.csv`
- cache scaling：`benchmark/gptq_cache_scaling/{ar,pard2_ti}_humaneval_cache{2048,4096,8192}.{json,log}` 與各自 `_amd_smi.csv`
- Kernel：`kernel/quarot_HIP_gfx1201_g4_m2.so` 與 sweep/correctness JSON、CSVs
- 正式 8K：`benchmark/gptq_formal_8k/` 下六組 JSON/log/AMD-SMI CSV 與 `ALL_BENCHMARKS_COMPLETE`
- Qualification：`accuracy/formal_8k/qualification.json`
- Accuracy：`accuracy/formal_8k/` 下三資料集 score/detail、HumanEval sandbox status、`SCORER_PREP_COMPLETE`；官方 parquet 固定於 `accuracy/ground_truth_sources/`
- TD proxy：`benchmark/td_proxy_smoke_2k/` 下三組JSON/log/AMD-SMI CSV；所有結果均為`qualified=false`、`target_alignment=cross_target_proxy`

## 10. HumanEval 2K / 4K / 8K cache scaling smoke

固定條件與第 7 節相同，唯一變數是 `--max-cache-len`；所有程序獨立啟動、以 1 秒 AMD SMI CSV 監測，完成後確認無 GPU process。8K 依使用者指示採 95% gate；實際 peak 也未超過一般 90% gate。

| cache | AR external peak | TI external peak | AR warmup max allocated | TI warmup max allocated | AR/TI parity |
|---:|---:|---:|---:|---:|---|
| 2K | 27,064 MiB（82.96%） | 28,201 MiB（86.44%） | 18,259.4 MiB | 19,539.5 MiB | 32/32 exact |
| 4K | 27,192 MiB（83.35%） | 28,347 MiB（86.89%） | 18,395.4 MiB | 19,919.7 MiB | 32/32 exact |
| 8K | 27,448 MiB（84.13%） | 28,619 MiB（87.72%） | 18,667.4 MiB | 20,691.3 MiB | 32/32 exact |

| cache | AR steady / E2E tok/s | TI steady / E2E tok/s | mean accept | draft / conditional acceptance |
|---:|---:|---:|---:|---:|
| 2K | 15.0930 / 14.1623 | 9.8222 / 8.7553 | 3.25 | 0.1500 / 0.6923 |
| 4K | 15.7090 / 14.6913 | 10.5473 / 9.3167 | 3.50 | 0.1667 / 0.7143 |
| 8K | 15.5302 / 14.5348 | 10.3880 / 9.0612 | 3.50 | 0.1667 / 0.7143 |

主要觀察：

- AR 每增加 2K cache，external peak 精確增加 128 MiB；4K→8K 增加 256 MiB，符合 target KV4 每 token 64 KiB 的理論值。
- TI external peak 2K→4K 增加 146 MiB、4K→8K 增加 272 MiB；但 PyTorch warmup `max_allocated` 2K→8K 增加 1,151.8 MiB，較接近 target KV4 加 draft BF16 KV 的實際配置增量。1 秒 AMD SMI peak 反映的是整體 concurrent peak，可能漏掉較短的 allocator 瞬時高點，不能單獨用來推算 cache bytes。
- warmup `max_reserved` 的 AR 2K/4K/8K 為 26,628/26,756/27,012 MiB；TI 為 27,670/27,816/28,088 MiB，均與 external peak 一致顯示 8K 可容納。
- 三個 cache 的 AR output IDs 完全相同，各 cache 的 TI 也與 paired AR 32/32 exact。2K 的 acceptance 統計略低於 4K/8K，但這是單 prompt、單 sweep，不能視為 cache 對 acceptance 的因果效應。
- 8K TI 距 95% gate 約 2,374 MiB；距一般 90% gate 約 743 MiB。可執行，但若改 batch、長 prompt、compile/autotune 或加入 TD warp，必須重新量測，不能沿用此 headroom。
- 最高溫度為 4K AR hotspot 66°C、8K TI memory 46°C；沒有 thermal safety event。4K AR CSV 有一次 312 W raw sample，略高於 `max_power=300 W`，保留為 AMD SMI sampling anomaly，不據此判定實體超功耗。

## 11. 完成後分析與下一階段建議

1. 目前 GPTQ checkpoint、kernel dimensions 與 8K memory budget 都已通過正式負載；沒有理由改動 QuaRot/PARD2 架構或 5120/8192/1024/25600 dispatch。仍應維持 sequential execution，因 TI 雖低於 95%，但改 batch、compile/autotune、draft dtype 或 TD 路徑都會改變 headroom。
2. 三資料集 TI 均有約 1.44–1.52× paired median speedup，mean accept 6.28–6.98；若下一階段追求速度，優先分析 acceptance/routing 與 draft 成本，而不是再調已驗證的 GEMM dimension。
3. MATH-500 多數輸出觸及 256-token 上限，conservative exact 僅 10%；下一個 accuracy 實驗應獨立提高 generation budget，並引入可稽核的數學等價 scorer，不能把本輪下界解讀成模型能力上限。
4. K1-on 必須先修復 fused RoPE+KV4 numerical contract 並重跑 AR/TI parity；adaptive routing、draft 量化或 TD cross-target proxy 仍屬需另行授權的架構實驗。

目前沒有未解的 VRAM、checkpoint correctness、parity 或 kernel-dimension blocker。正式長 benchmark 與獨立 scorer 已完成；experimental TD proxy 不在本階段範圍。

## 12. 8K 正式 benchmark 與獨立 task scorer 契約

依 2026-08-27 使用者指示，本輪正式測試固定為 HumanEval 80、GSM8K 80、MATH-500 20 prompts，各跑 3 sweeps；AR 與 PARD2-TI 皆為 batch 1、greedy、正常 EOS、最多 256 generated tokens、8 warmups、cache 8192、eager、K1-off、waves 4/2。所有 GPU run sequential execution，8K VRAM hard gate 為 95%；另以 hotspot 110°C、memory temperature 108°C 作安全停止線。TI draft、target checkpoint、tokenizer、HIP SHA 與前述 smoke 完全相同，因此不在正式測試中變更 QuaRot/PARD2 架構或 kernel dimension。

每個 run 同時保留：benchmark JSON、stdout/stderr log、1 秒 AMD-SMI CSV，以及回填至 JSON 的 `external_vram_monitor` 摘要。外部摘要包含 CSV SHA256、樣本數、時間範圍、VRAM MiB/百分比峰值、hotspot/memory 溫度與原始 power/gfx/mem peaks；回填工具會以 GPU total bytes 反算百分比，若與 AMD-SMI 欄位差超過 0.15 percentage point 即 fail-close。正式 runner 在 VRAM `>=95%` 或上述任一溫度線時送出 SIGINT 並留下 safety-gate log。

### 12.1 Ground truth provenance 與對齊

本地 PARD benchmark prompt 檔不包含 HumanEval tests，GSM8K prompt 也不含答案，因此 task scorer 另以 Hugging Face Dataset Viewer API 取得官方 parquet；沒有用模型輸出、網路搜尋摘要或手工答案當 ground truth。

| source | split / 用途 | SHA256 |
|---|---|---|
| `openai/openai_humaneval` | test；canonical test/check 與 entry point | `2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6` |
| `openai/gsm8k` | main/test；答案與 split 對齊 | `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` |
| `openai/gsm8k` | main/train；確認 prompt 未誤落 train | `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d` |

GSM8K 本地 80/80 questions 與官方 test split exact text match，且沒有 train match。HumanEval 因本地 prompt 包裝文字不同，使用 top-level function entry point 對齊，80/80 entry points 唯一且全部找到官方 task；candidate 會和官方 `test` 合併並呼叫 `check(entry_point)`。MATH-500 的本地固定 JSONL 已包含 `unique_id`、solution 與 answer，scorer 直接使用 pinned answer。

### 12.2 Accuracy 定義

- GSM8K：移除 thinking 區塊後，依 `\\boxed{}`、final-answer/`####` marker、最後數值的優先序抽取；以 `Decimal`/`Fraction` 正規化後 numeric exact match。
- MATH-500：使用最後一個可平衡解析的 `\\boxed{}`，否則 final-answer marker 或最後非空行；只做空白、`\\left`/`\\right`、spacing command、`dfrac/tfrac` 等保守正規化。僅在兩側都是純 numeric token 時允許數值等價，不使用 CAS 或寬鬆語義判定，因此可能低估等價但不同形式的答案，不會把未證實等價算成正確。
- HumanEval：每個 sweep 都是獨立 pass@1 candidate；先選取包含正確 entry point 的 Python fence/function，否則把 completion 接到 canonical prompt，再以官方 tests 執行。`correct` 僅在 sandbox 未 timeout 且 exit code 0 時成立。

除 aggregate accuracy 外，scorer 保存每個 prompt/sweep 的 decoded output、prediction/expected、判定與 alignment，並分 sweep 報告 accuracy；另記錄同一 prompt 三 sweep 的 token sequence 是否完全一致。benchmark timing 與 scorer 執行時間完全分離。

### 12.3 HumanEval sandbox

模型產生的程式不在 benchmark 容器或 host 直接執行。每個 candidate 使用本機固定 image `python:3.11-alpine`（image ID `sha256:a86c86cb5c6bdf7fbb2bcf7bdc0a68b40e46e41d5a340067b0f82679b7d62713`）建立獨立暫時容器，固定 `--network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 32 --memory 128m --memory-swap 128m --cpus 1`，candidate 以唯讀 bind mount 掛載，`/tmp` 為 16 MiB noexec tmpfs。外層 timeout 預設 10 秒，以唯一容器名重試清理 Docker create/timeout race；pass、assertion failure、infinite-loop timeout smoke 均已驗證，修復後沒有殘留容器。

六個 GPU run 與三個 scorer 均已完成；正式數值、qualification 與可重現限制列於下一節。

## 13. 8K 正式結果

### 13.1 AR HumanEval

AR HumanEval 已成功完成並通過獨立 artifact audit：80 prompts × 3 sweeps = 240 runs；每個 prompt 的 sweep index 精確為 `{0,1,2}`，沒有缺漏或重複。三個 sweeps 對 80/80 prompts 的 output-token sequences 都完全相同。正常 EOS 下共生成 60,471 tokens，每 run 166–256 tokens。

| 項目 | 結果 |
|---|---:|
| median steady throughput | 15.9713 tok/s |
| median end-to-end throughput | 15.6679 tok/s |
| external VRAM peak | 27,448 MiB（84.13%） |
| 95% gate | pass；headroom 10.87 percentage points |
| hotspot / memory temperature peak | 98°C / 84°C |
| AMD-SMI samples | 4,039 |
| AMD-SMI CSV SHA256 | `91a02db966e3af878ca56909155a99f742850163ac0298ae957a44fef2b8851a` |
| benchmark JSON SHA256 | `0a020eb07f7177a1ad154d0e47d007483388b054f02e45fb5cbb38f948a10094` |

外部監測涵蓋 timestamp 1787814062–1787818211（69 分 9 秒），沒有 VRAM/thermal safety event。`power_usage` raw peak 為 376 W，高於裝置報告的 300 W 上限，依第 4.2 節既有判定只作 AMD-SMI raw sampling anomaly 保存，不當成可信實體功耗；VRAM、hotspot 與 memory 欄位仍自洽且用於 gate。

### 13.2 AR GSM8K

AR GSM8K artifact audit 同樣通過：80 prompts × 3 sweeps = 240 runs，每個 prompt 的 sweep index 完整且 80/80 prompts 的三 sweep output-token sequences 完全相同。正常 EOS 下共生成 54,777 tokens，每 run 102–256 tokens。

| 項目 | 結果 |
|---|---:|
| median steady throughput | 16.2215 tok/s |
| median end-to-end throughput | 16.0687 tok/s |
| external VRAM peak | 27,448 MiB（84.13%） |
| 95% gate | pass；headroom 10.87 percentage points |
| hotspot / memory temperature peak | 95°C / 84°C |
| AMD-SMI samples | 3,583 |
| AMD-SMI CSV SHA256 | `4374026861afc4b40bccd90340506677e1b01e35a2b6d3904c135ee6224278b3` |
| benchmark JSON SHA256 | `59d7eea05780ec09ffb5e43434e3efadcc54bb55c1a271f5d5ba968ee5129fff` |

外部監測 timestamp 1787818213–1787821889，約 61 分 16 秒，沒有 VRAM/thermal safety event。AMD-SMI `power_usage` raw peak 378 W 仍依前述 anomaly policy 保存。

### 13.3 AR MATH-500

AR MATH-500 artifact audit 通過：20 prompts × 3 sweeps = 60 runs，每個 prompt 的 sweep index 完整，20/20 prompts 的三 sweep output-token sequences 完全相同。正常 EOS 下共生成 15,330 tokens，每 run 250–256 tokens。

| 項目 | 結果 |
|---|---:|
| median steady throughput | 16.0558 tok/s |
| median end-to-end throughput | 15.8984 tok/s |
| external VRAM peak | 27,448 MiB（84.13%） |
| 95% gate | pass；headroom 10.87 percentage points |
| hotspot / memory temperature peak | 100°C / 84°C |
| AMD-SMI samples | 1,172 |
| AMD-SMI CSV SHA256 | `f1f4d6a9084fb188ff172ffe8113a4bd07964bce7253843dc6e5ffbf3e94d999` |
| benchmark JSON SHA256 | `45c2ffa14f5f0082d9e1cc8670e42ebec3ffa842ac179283e57398004007f403` |

外部監測 timestamp 1787821891–1787823079，約 19 分 48 秒，沒有 VRAM/thermal safety event。AMD-SMI `power_usage` raw peak 377 W 依 anomaly policy 保存。至此三個 AR 正式 payload 共 540 runs，prompt/sweep coverage、cross-sweep determinism 與 95% VRAM gate 全部通過。

### 13.4 PARD2-TI HumanEval

TI HumanEval formal-payload validation 與 artifact audit 通過：240 runs、完整 80×3 coverage，且與 paired AR 的 240/240 output-token sequences exact parity。速度判定使用相同 prompt/sweep 的 `TI steady tok/s ÷ AR steady tok/s`，不是兩個 aggregate median 相除。

| 項目 | 結果 |
|---|---:|
| TI median steady throughput | 24.2617 tok/s |
| TI median end-to-end throughput | 23.0743 tok/s |
| paired median steady speedup | 1.5185× |
| bootstrap 95% speedup CI | [1.4537, 1.5673] |
| run-level speedup CV | 0.8859% |
| exact AR parity | 240/240 |
| mean accept length | 6.7569 |
| mean draft acceptance | 0.3838 |
| mean conditional acceptance | 0.8646 |
| external VRAM peak | 28,619 MiB（87.72%） |
| hotspot / memory temperature peak | 102°C / 80°C |
| HumanEval hard gate | pass |
| AMD-SMI samples / SHA256 | 2,821 / `5919ed2278f88dd4b9f6c88c6e471779fb67b94fc91fb85cecf879f8902b3a9c` |
| benchmark JSON SHA256 | `91a2904cb66383674c5ddaaefbf9cea1d5c560d33b1547c1003ac4d736efa512` |

外部監測 timestamp 1787823079–1787825965，約 48 分 6 秒；VRAM 距 95% gate 7.28 percentage points，hotspot 距 110°C 停止線 8°C，沒有 safety event。`power_usage` raw peak 335 W 仍依 anomaly policy 保存。HumanEval 的 1.52× speedup 與 6.76 mean accept 顯示完整 80-prompt acceptance 遠高於先前單 prompt smoke（mean accept 3.5、steady 0.67×）；這也證明不能以單 prompt smoke 決定 PARD2 是否有整體效益。

### 13.5 PARD2-TI GSM8K

TI GSM8K formal-payload validation 與 artifact audit 通過：240 runs、完整 80×3 coverage，與 paired AR 的 240/240 output-token sequences exact parity。

| 項目 | 結果 |
|---|---:|
| TI median steady throughput | 23.4642 tok/s |
| TI median end-to-end throughput | 22.5696 tok/s |
| paired median steady speedup | 1.4448× |
| bootstrap 95% speedup CI | [1.4167, 1.4709] |
| run-level speedup CV | 0.0502% |
| exact AR parity | 240/240 |
| mean accept length | 6.2799 |
| mean draft acceptance | 0.3520 |
| mean conditional acceptance | 0.8490 |
| external VRAM peak | 28,619 MiB（87.72%） |
| hotspot / memory temperature peak | 94°C / 80°C |
| GSM8K hard gate | pass |
| AMD-SMI samples / SHA256 | 2,643 / `7553477ca463518387197820c099f0894214e3207bb04336c92bc2c1b0450ae1` |
| benchmark JSON SHA256 | `d5c49f1283e325df88925ae6f25aee3f7471efb0cff820a789e10c27592b83e1` |

外部監測 timestamp 1787825966–1787828668，約 45 分 2 秒，沒有 VRAM/thermal safety event；`power_usage` raw peak 336 W 依 anomaly policy 保存。完整樣本的 1.44× speedup、6.28 mean accept 與極低 CV 再次顯示 TI 在此 target/draft 組合上有穩定效益。

### 13.6 PARD2-TI MATH-500

TI MATH-500 formal-payload validation 與 artifact audit 通過：60 runs、完整 20×3 coverage，與 paired AR 的 60/60 output-token sequences exact parity。

| 項目 | 結果 |
|---|---:|
| TI median steady throughput | 23.3538 tok/s |
| TI median end-to-end throughput | 23.0159 tok/s |
| paired median steady speedup | 1.4505× |
| bootstrap 95% speedup CI | [1.2692, 1.6647] |
| run-level speedup CV | 0.4292% |
| exact AR parity | 60/60 |
| mean accept length | 6.9778 |
| mean draft acceptance | 0.3985 |
| mean conditional acceptance | 0.8565 |
| external VRAM peak | 28,619 MiB（87.72%） |
| hotspot / memory temperature peak | 100°C / 80°C |
| MATH-500 hard gate | pass |
| AMD-SMI samples / SHA256 | 886 / `7062622eaefd152f68020f97245b20c29b9f63e7ba7c19aee0dc8bdc8f1aa840` |
| benchmark JSON SHA256 | `980a504f99f2bdd29cedd6f40cd7b2ac21ab0f68c124dfb67ec8e91c325dd6a0` |

外部監測 timestamp 1787828669–1787829565，約 14 分 56 秒，沒有 VRAM/thermal safety event。至此六組正式 payload 全部完成，AR/TI 共 1,080 runs。

## 14. 最終 qualification 與獨立 accuracy

### 14.1 PARD2 qualification

| dataset | paired speedup | 95% CI | CV | AR parity | TI VRAM | hard gate |
|---|---:|---:|---:|---:|---:|---|
| HumanEval | 1.5185× | [1.4537, 1.5673] | 0.8859% | 240/240 | 28,619 MiB（87.72%） | pass |
| GSM8K | 1.4448× | [1.4167, 1.4709] | 0.0502% | 240/240 | 28,619 MiB（87.72%） | pass |
| MATH-500 | 1.4505× | [1.2692, 1.6647] | 0.4292% | 60/60 | 28,619 MiB（87.72%） | pass |

`qualification.json` 的 overall `hard_gate=true`；canonical model 為 Qwen3-32B，VRAM gate 明確記錄為 95%，SHA256 `a2441cd1b4837e8d7b7705a6a92f78853e92e7443ee2acb337d824c7614372ae`。六組 run 皆無 safety-gate log；正式最高 hotspot 102°C、最高 memory temperature 84°C。

### 14.2 獨立 task scorer

| dataset | 每 sweep | aggregate | metric | cross-sweep token exact |
|---|---:|---:|---|---|
| HumanEval | 40/80 | 120/240 = 50.00% | isolated official-test execution pass@1 | 80/80 |
| GSM8K | 47/80 | 141/240 = 58.75% | numeric exact | 80/80 |
| MATH-500 | 2/20 | 6/60 = 10.00% | conservative normalized exact | 20/20 |

HumanEval 240 candidates 全數在第 12.3 節 sandbox 執行，120 pass、0 timeout；status SHA256 `a257f64b5bfb5d450b9e25c2dfffaa82f46a233ab7f4831f643b3028e2aa9694`，score SHA256 `c38a95fd8636c36276d5ba34765e2a0a0908204e84e848aada08b44a676da692`。第一次 host invocation 誤把 bind root 當 repo root，於執行 candidate 前即失敗；修正為含 `/fused_v1` 的路徑後完整成功，未留下 sandbox container。

GSM8K score SHA256 `484874b8fc0bdb07e26fcb767ed331d1c670549d08f2b3850ff903ef5ff81349`；MATH-500 score SHA256 `2e5ed1204fe2cf61da6862c29441aead8db496cdb8900ec70193ca76b44a102b`。三 sweep 結果相同是 greedy token determinism 的直接結果，不代表 240 個統計獨立樣本。MATH-500 60 runs 中生成長度為 250–256 tokens，且大多觸及上限；加上 scorer 不使用 CAS，10% 應視為保守、可重現的下界。

## 15. Qwen3-14B-on-Qwen3-32B experimental TD proxy

2026-08-28依序測試pinned `amd/PARD2-Qwen3-14B@679eff0b65ffaf5abd2dadd21a17909562935798`。兩個PARD2 Qwen checkpoint的drafter backbone皆為28 layers、hidden 1024、約0.8B；差別是8B warp輸入`4×4096=16384`，14B warp為`4×5120=20480`。8B warp與32B hidden width硬性不相容；14B warp可通過shape契約，但因40-layer Qwen3-14B與64-layer Qwen3-32B在相同taps `[-1,-8,-16,-24]`上語意不同，只能使用明確的`qwen3-14b-on-qwen3-32b` cross-target proxy profile。

### 15.1 2K / 32-token smoke結果

| dataset | AR steady | TD steady | TD/AR | TD/TI smoke | mean accept | accepted/proposed | parity | external VRAM |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| HumanEval | 15.8252 | 3.5768 | 0.2260× | 0.3462× | 1.0 | 0/480 | 32/32 exact | 28,259 MiB（86.62%） |
| GSM8K | 15.7974 | 3.5691 | 0.2259× | 0.2741× | 1.0 | 0/480 | 32/32 exact | 28,259 MiB（86.62%） |
| MATH-500 | 15.4991 | 3.5825 | 0.2311× | 0.2203× | 1.0 | 0/480 | 32/32 exact | 28,259 MiB（86.62%） |

三個run各有32 verifier steps：每一步提出15個draft tokens後全部被拒絕，只由target發出1 token。Target verify stage約7.84–7.86秒，draft/projection path約1.07–1.09秒；這是speculative decoding的最差工作型態。結果仍為lossless，因verifier正確回退至AR token，但速度只有AR約23%。

| dataset | JSON SHA256 | AMD-SMI CSV SHA256 | hotspot / memory peak |
|---|---|---|---:|
| HumanEval | `fd074a606bb10b7f6bfeb082de9b0678fcab50463e0ce447e058177016827af9` | `dc953ae898271ed616b47c8719effd0e77bc8542dd1f21347f22338464aef979` | 56°C / 44°C |
| GSM8K | `5758479f842810193b97e172874ad62dea96bc80081a424bedbd0fcb72cc526a` | `9c7b7c2656f2d5a8683cb79b7ae93aa2a97ffb9908f9bb0754eefd03f93bdf13` | 58°C / 46°C |
| MATH-500 | `400485d7051c3f640e0533de864e6d34103751eceb952605c3ea29f439bddf5d` | `14abaff382497c0c6a0b036e778f3253e0b93b5b690c111e086262863060fc61` | 59°C / 48°C |

### 15.2 判定

- CPU preflight通過：warp shape `(1024,20480)`、HadK signs/final norm `(5120,)`、pinned revisions與target provenance均正確；TD proxy shape/revision/basis/metadata focused regression 9/9 passed。
- 三組`target_alignment=cross_target_proxy`、`qualification_track=experimental`、`qualified=false`，不會混入canonical qualification。
- 95% VRAM與110/108°C thermal gates全部通過，無OOM、ROCm fault或殘留GPU process；容量不是失敗原因。
- 三資料集完全相同的zero-accept顯示主因是14B warp/drafter與32B hidden-feature分布不對齊，而非單一task、output parity或kernel shape。

因此依預先設定的acceptance/performance gate取消8K/256-token matched sample與完整formal。繼續長跑只會重複全拒絕路徑，無法提供值得用GPU時間交換的新資訊。

### 15.3 Drafter選擇結論

| 選項 | 32B TD判定 |
|---|---|
| PARD2-Qwen3-8B | TI可用；TD projection width 16384與32B所需20480不相容，拒絕 |
| PARD2-Qwen3-14B | shape可用但實測zero-accept，只能作已被否證的cross-target proxy，拒絕進formal |
| 32B-aligned PARD2 | 正確方向：可沿用0.8B/28-layer draft架構，但須以Qwen3-32B teacher、選定的32B taps與GPTQ target numerics重新訓練/校準20480→1024 warp及draft |

下一步不是調整現有14B warp的layer index或直接重跑8K，而是建立Qwen3-32B-aligned TD checkpoint。若只需可部署路徑，維持已qualified的TI；若要研究TD，應先在小型calibration/tune split證明mean accept顯著高於TI並通過AR parity，再進正式benchmark。
