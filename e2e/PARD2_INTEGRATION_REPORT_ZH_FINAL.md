# fused_v1 × PARD2-TI/TD 整合與 Qualification 報告

> 最終狀態：Qwen3-8B、batch 1、greedy、W4A4KV4 target 的第一階段整合已完成；HumanEval、GSM8K、MATH-500 全部 exact parity，TI/TD 全部通過速度、穩定性與 VRAM 硬門檻。TD 尚未達文獻速度的 70% stretch target。

## 1. 範圍與資源

原定 Llama 3.1 8B 因 fused_v1 支援與已驗證程度有限，改用 dense Qwen3-8B。MoE、sampling、batch serving、模型重訓與 Qwen PARD2 以外模型不在首版範圍。

- Target：`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`，fused_v1 RTN W4A4KV4 checkpoint。
- Drafter：`amd/PARD2-Qwen3-8B@67a1516c8f6fc145cda99916799a0cbb3a4af135`。
- 上游程式：AMD PARD commit `6f279bf3f1680e0b5d71c562ca5b91bdeef4c038`。
- 固定規格：`draft_k=15`、PARD token `151670`、TD taps `[-1,-8,-16,-24]`、target dim `16384`、projection scale `0.02`。
- 平台：gfx1201 32 GiB、ROCm 7.2；只比較同硬體 speedup，不把絕對 TPS 與論文 A100 數字等同。

## 2. 架構

AR、TI、TD 共用同一 target checkpoint、量化語意、attention/KV kernel 與 cache contract。每輪 speculative step 由 pending token 加最多 15 candidates 組成 q_len≤16 的 verifier chunk；接受最長相同 prefix，再提交 target correction/bonus token。

### 2.1 KV、GQA 與 transaction

- KV cache 永久保存 Qwen3 原生 8 KV heads，不展開成 32 heads；expanded-MHA 保留為 correctness oracle。
- 新增 asymmetric KV4 native-GQA append/decode，以及 q_len 1/15/16 的 fused multi-token append。
- K 在 Hadamard/rotation basis，V 保持原 basis，與 decode kernel contract 一致。
- virtual causal metadata 讓 chunk 內 token 只能看見合法 prefix；`begin/commit/rollback` 只保留 accepted prefix 加 correction，stale slots 留待覆寫。

### 2.2 Small-chunk verifier

- q_len 2..16 走專用 small-chunk 路徑，避免誤入大型 prefill。
- 增加 M=16 W4A4 GEMM dispatch、execution-time Q/K/V 與 gate/up projection grouping，以及 fused SiLU–Hadamard–quantize–down FFN。
- checkpoint keys 保持相容，不做 migration；TI 不載入 warp projection，TD 只保存四個指定 hidden taps。
- TD target features 左移一格，首位補零，再套官方 projection。

### 2.3 Drafter 與執行

官方 BF16 drafter 使用固定 shape 與 `torch.compile(max-autotune)`；compile budget 覆蓋 k+1。candidate、logit、feature、metadata 與 transaction buffers 預配置，降低每輪 allocation 與同步。TI 與 TD 使用獨立、可持久化 Inductor cache。

## 3. Exact parity 問題與修復

早期 MATH prompt 14 在 token 171 出現 TI/TD 同步偏離。expanded-MHA、關閉 fused append 都不能消除問題；fresh sequential cache 可恢復 parity。逐層比對定位到 full-accept chunk 後 layer 8/9 MLP/KV 狀態的微小差異。

根因不是 PARD 接受規則或 GQA cache，而是 PyTorch RMSNorm reduction 隨 M/row grouping 改變浮點加總順序；微小 rounding 會跨過 activation/KV4 量化邊界並累積。M4/M8 grouping 仍不可靠，逐 row M1 正確但過慢。

最終新增 HIP row-independent RMSNorm：每 row 一個 workgroup，使 M=16 與 16 次 M1 bit-exact。所有 target input/post-attention/final norm 在 exact small-chunk contract 下共用此 kernel；LM head 保留 rowwise M1。新 AR/TI/TD 因此有共同數值契約。它與舊 PyTorch AR 數學等價，但極少數 rounding/token 可能不同，正式 parity 定義以新共同契約為準。

## 4. 測試與 Benchmark 方法

- GPU correctness：native GQA 對 expanded oracle、chunk verifier 對 sequential AR、accept length 0..15 後 cache 對 fresh recomputation。
- 正式資料：HumanEval 80、GSM8K 80、MATH-500 20；每 prompt 256 generated tokens、batch 1、8 warmups、3 個交錯 measured sweeps。
- 每 mode/dataset 獨立程序；preexisting VRAM >1 GiB 即拒絕執行。
- 指標：steady/E2E tok/s、TTFT/stage、accept length、conditional acceptance、peak VRAM、paired bootstrap 95% CI、sweep-median CV。
- 最終 focused suite：`110 passed`，0 failure；4 個 Triton deprecation warnings。

## 5. 正式結果

| Dataset | Mode | steady tok/s | E2E tok/s | paired speedup | 95% CI | CV | mean accept | parity | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HumanEval | AR | 20.914 | 20.803 | 1.000× | — | 0.07% | — | reference | 6.05 GiB |
| HumanEval | TI | 34.770 | 32.344 | 1.667× | [1.620,1.731] | 0.94% | 5.968 | 240/240 | 8.62 GiB |
| HumanEval | TD | 39.928 | 37.037 | 1.921× | [1.843,1.998] | 0.58% | 6.696 | 240/240 | 8.71 GiB |
| GSM8K | AR | 13.569 | 13.515 | 1.000× | — | — | — | reference | — |
| GSM8K | TI | 33.636 | 31.637 | 2.361× | [2.150,2.757] | 0.23% | 5.450 | 240/240 | 8.20 GiB |
| GSM8K | TD | 39.244 | 36.604 | 2.802× | [2.559,3.193] | 0.50% | 6.373 | 240/240 | 8.26 GiB |
| MATH-500 | AR | 21.102 | 20.947 | 1.000× | — | 0.41% | — | reference | 6.04 GiB |
| MATH-500 | TI | 33.361 | 31.371 | 1.586× | [1.479,1.644] | 1.10% | 5.536 | 60/60 | 8.53 GiB |
| MATH-500 | TD | 36.685 | 35.276 | 1.743× | [1.700,2.048] | 0.34% | 6.517 | 60/60 | 8.61 GiB |

Qualifier 的 paired speedup 是 prompt-level median，因此與表中 aggregate steady TPS 直接相除可能略有不同。三資料集 TI/TD CI 下界皆 >1、CV <5%、保留至少 10% VRAM，且 exact greedy parity 全部通過；第一階段 hard gate 為 PASS。

文獻 stretch：TD speed fraction 為 HumanEval 28.5%、GSM8K 43.5%，均未達第一階段 70%；MATH 沒有對應 Qwen 文獻值，不虛構比例。結果證明在 gfx1201 上已穩定優於 AR，但尚未趨近論文 A100 實作。

## 6. 實驗與採用決策

- expanded-MHA 與 non-fused append 診斷排除 GQA/cache kernel 為 parity 根因，最終仍採 native GQA。
- M16 dispatch 的局部 steady 改善不足以保證 E2E；只保留通過 parity 與整體收益的預設組合。
- feature calibration 曾改善局部 steady，但 E2E 無淨收益，未納入第一階段。
- drafter W4A4 candidate agreement/acceptance 不達門檻，未採用。
- adaptive-k 與 graph capture 未加入第一階段，避免把額外元素混入核心架構結論。

## 7. 第二階段

後續依序：以 fused target tune split 重擬 raw/projected per-channel affine calibration；將 drafter 移植 W4A4 並要求 agreement≥95%、accept drop≤5% 且 E2E 上升；在 k={8,12,15} 以 tune split 固定 confidence/EMA adaptive-k；最後評估 fixed-shape HIP graph capture，改善不足 3% 不採用。formal split 不再調參，逐項與組合 ablation。

## 8. 主要檔案與重現

- Runtime/CLI：`e2e/speculative.py`、`e2e/pard2.py`、`e2e/benchmark_pard2.py`。
- Qualification：`e2e/qualify_pard2.py`；結果位於 `pard2_formal_results_hip_norm/qualification_phase1.json`。
- Cache/kernel：`quarot/transformers/kv_cache.py`、`quarot/kernels/fused_hip.hip`、`quarot/kernels/include_hip/fused.h`、`quarot/kernels/bindings.cpp`。
- Tests：`tests/test_speculative.py`、`tests/test_fused_hip.py`、`tests/test_pard2_gpu.py`、`tests/test_model_compat.py`。

## 9. 結論

第一階段已完成可重現、exact-parity 且在三資料集均快於 AR 的 PARD2-TI/TD 整合。最關鍵的工程成果是 native GQA KV4 transaction verifier，以及消除 chunk-size-dependent rounding 的 row-independent HIP RMSNorm。TD 為三資料集最快模式，paired speedup 1.74×–2.80×；下一階段的核心目標不是再修 correctness，而是降低 verifier/drafter overhead、提高 acceptance 利用率，縮小與文獻效率的差距。
