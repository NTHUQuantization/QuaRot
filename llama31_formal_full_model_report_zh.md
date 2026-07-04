# Llama-3.1 8B Formal QuaRot Wrapper Full-Model 評估

## 範圍

- 使用 `meta-llama/Llama-3.1-8B` 真實權重與 tokenizer。
- 新增 `QuaRotLlamaForCausalLM` formal token-by-token wrapper，提供 `prefill()`、`decode_one()`、`generate()`。
- Prefill 仍使用 HF Llama/SDPA，並用 `logits_to_keep=1` 避免長 context 產生整段 vocab logits。
- `quarot_unfused` / `fused_quarot` 的 decode token 經過 GQA-aware K1/K2/K3/FFN path；不是 monkey-patch transformers `generate()` internals。
- QuaRot 權重旋轉與 calibration 尚未完成，因此品質結果只能視為目前 prototype path 的輸出差異。

## Latency 設定

- batch = `1,2,4,8`
- context length = `10,128,1024,4096`
- warmup = `1`, iters = `3`, repeats = `3`
- attention implementation = `sdpa`
- prefill 包含 prompt forward 與 cache 建立；QuaRot modes 的 prefill 另包含 HF KV cache 轉 paged INT4 cache。

## Decode Latency / Speedup

| batch | context_len | fp16_hf_ms | quarot_unfused_ms | fused_quarot_ms | fused_vs_fp16 | fused_vs_unfused |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 38.43 | 117.3 | 40.69 | 0.9445 | 2.884 |
| 1 | 128 | 38.04 | 116.2 | 41.88 | 0.9082 | 2.773 |
| 1 | 1024 | 40.54 | 112.3 | 43.93 | 0.9227 | 2.557 |
| 1 | 4096 | 44.82 | 117.2 | 49.52 | 0.905 | 2.366 |
| 2 | 10 | 39.88 | 134.4 | 42.15 | 0.9461 | 3.188 |
| 2 | 128 | 38.83 | 118.4 | 41.59 | 0.9336 | 2.845 |
| 2 | 1024 | 40.98 | 117.1 | 44.41 | 0.9228 | 2.637 |
| 2 | 4096 | 48.14 | 133.3 | 49.3 | 0.9765 | 2.703 |
| 4 | 10 | 38.8 | 133.8 | 41.99 | 0.9239 | 3.186 |
| 4 | 128 | 41.31 | 119.1 | 42.55 | 0.9708 | 2.8 |
| 4 | 1024 | 43.73 | 123.2 | 44.29 | 0.9875 | 2.782 |
| 4 | 4096 | 55.87 | 184.3 | 50.09 | 1.115 | 3.679 |
| 8 | 10 | 39.37 | 116.6 | 43.01 | 0.9154 | 2.711 |
| 8 | 128 | 41.1 | 132.1 | 43.24 | 0.9504 | 3.056 |
| 8 | 1024 | 47.31 | 133 | 45.22 | 1.046 | 2.941 |
| 8 | 4096 | 71.19 | 298.3 | 52.58 | 1.354 | 5.673 |

## Full Latency Summary

| mode | batch | context_len | metric | mean | median | std | p90 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16_hf | 1 | 10 | decode_ms_per_token | 38.43 | 38.37 | 0.8497 | 39.13 | 39.22 |
| fused_quarot | 1 | 10 | decode_ms_per_token | 40.69 | 40.67 | 0.09382 | 40.77 | 40.78 |
| quarot_unfused | 1 | 10 | decode_ms_per_token | 117.3 | 114.6 | 5.855 | 122.2 | 123.1 |
| fp16_hf | 1 | 128 | decode_ms_per_token | 38.04 | 38.02 | 0.04252 | 38.07 | 38.08 |
| fused_quarot | 1 | 128 | decode_ms_per_token | 41.88 | 40.88 | 1.898 | 43.43 | 43.75 |
| quarot_unfused | 1 | 128 | decode_ms_per_token | 116.2 | 116.6 | 3.871 | 119.2 | 119.5 |
| fp16_hf | 1 | 1024 | decode_ms_per_token | 40.54 | 40.54 | 0.1076 | 40.62 | 40.63 |
| fused_quarot | 1 | 1024 | decode_ms_per_token | 43.93 | 42.89 | 1.908 | 45.49 | 45.81 |
| quarot_unfused | 1 | 1024 | decode_ms_per_token | 112.3 | 112.3 | 0.6927 | 112.9 | 113 |
| fp16_hf | 1 | 4096 | decode_ms_per_token | 44.82 | 44.74 | 0.1345 | 44.92 | 44.95 |
| fused_quarot | 1 | 4096 | decode_ms_per_token | 49.52 | 48.82 | 1.24 | 50.52 | 50.74 |
| quarot_unfused | 1 | 4096 | decode_ms_per_token | 117.2 | 118.5 | 2.513 | 118.7 | 118.7 |
| fp16_hf | 2 | 10 | decode_ms_per_token | 39.88 | 38.33 | 2.845 | 42.19 | 42.68 |
| fused_quarot | 2 | 10 | decode_ms_per_token | 42.15 | 41.19 | 1.733 | 43.56 | 43.86 |
| quarot_unfused | 2 | 10 | decode_ms_per_token | 134.4 | 134 | 12.49 | 144.4 | 145.7 |
| fp16_hf | 2 | 128 | decode_ms_per_token | 38.83 | 38.87 | 0.09828 | 38.9 | 38.9 |
| fused_quarot | 2 | 128 | decode_ms_per_token | 41.59 | 41.58 | 0.1061 | 41.68 | 41.69 |
| quarot_unfused | 2 | 128 | decode_ms_per_token | 118.4 | 119.5 | 6.882 | 123.6 | 124.1 |
| fp16_hf | 2 | 1024 | decode_ms_per_token | 40.98 | 40.98 | 0.162 | 41.11 | 41.12 |
| fused_quarot | 2 | 1024 | decode_ms_per_token | 44.41 | 43.99 | 1.342 | 45.52 | 45.72 |
| quarot_unfused | 2 | 1024 | decode_ms_per_token | 117.1 | 118.4 | 2.958 | 119 | 119.1 |
| fp16_hf | 2 | 4096 | decode_ms_per_token | 48.14 | 48.18 | 0.3206 | 48.38 | 48.41 |
| fused_quarot | 2 | 4096 | decode_ms_per_token | 49.3 | 49.28 | 0.1799 | 49.44 | 49.46 |
| quarot_unfused | 2 | 4096 | decode_ms_per_token | 133.3 | 133.1 | 0.2966 | 133.5 | 133.5 |
| fp16_hf | 4 | 10 | decode_ms_per_token | 38.8 | 38.81 | 0.1818 | 38.94 | 38.96 |
| fused_quarot | 4 | 10 | decode_ms_per_token | 41.99 | 41.93 | 0.1255 | 42.1 | 42.12 |
| quarot_unfused | 4 | 10 | decode_ms_per_token | 133.8 | 133 | 9.185 | 141.3 | 142.3 |
| fp16_hf | 4 | 128 | decode_ms_per_token | 41.31 | 39.41 | 3.318 | 43.99 | 44.57 |
| fused_quarot | 4 | 128 | decode_ms_per_token | 42.55 | 42.11 | 0.7861 | 43.19 | 43.32 |
| quarot_unfused | 4 | 128 | decode_ms_per_token | 119.1 | 120 | 3.447 | 121.6 | 121.8 |
| fp16_hf | 4 | 1024 | decode_ms_per_token | 43.73 | 42.29 | 2.597 | 45.84 | 46.29 |
| fused_quarot | 4 | 1024 | decode_ms_per_token | 44.29 | 43.96 | 0.6682 | 44.84 | 44.95 |
| quarot_unfused | 4 | 1024 | decode_ms_per_token | 123.2 | 119.6 | 9.944 | 131.5 | 132.9 |
| fp16_hf | 4 | 4096 | decode_ms_per_token | 55.87 | 54.76 | 2.119 | 57.6 | 57.96 |
| fused_quarot | 4 | 4096 | decode_ms_per_token | 50.09 | 50.2 | 0.1894 | 50.2 | 50.2 |
| quarot_unfused | 4 | 4096 | decode_ms_per_token | 184.3 | 184.4 | 0.1066 | 184.4 | 184.4 |
| fp16_hf | 8 | 10 | decode_ms_per_token | 39.37 | 39.32 | 0.107 | 39.46 | 39.48 |
| fused_quarot | 8 | 10 | decode_ms_per_token | 43.01 | 42.84 | 0.3223 | 43.27 | 43.33 |
| quarot_unfused | 8 | 10 | decode_ms_per_token | 116.6 | 115.4 | 4.871 | 120.6 | 121.3 |
| fp16_hf | 8 | 128 | decode_ms_per_token | 41.1 | 41.11 | 0.0368 | 41.12 | 41.13 |
| fused_quarot | 8 | 128 | decode_ms_per_token | 43.24 | 43.24 | 0.0185 | 43.26 | 43.26 |
| quarot_unfused | 8 | 128 | decode_ms_per_token | 132.1 | 131.9 | 20.88 | 148.9 | 151 |
| fp16_hf | 8 | 1024 | decode_ms_per_token | 47.31 | 47.28 | 0.1011 | 47.39 | 47.41 |
| fused_quarot | 8 | 1024 | decode_ms_per_token | 45.22 | 45.2 | 0.0513 | 45.26 | 45.27 |
| quarot_unfused | 8 | 1024 | decode_ms_per_token | 133 | 132.6 | 0.883 | 133.7 | 133.9 |
| fp16_hf | 8 | 4096 | decode_ms_per_token | 71.19 | 71.24 | 0.09051 | 71.24 | 71.25 |
| fused_quarot | 8 | 4096 | decode_ms_per_token | 52.58 | 52.52 | 0.1097 | 52.67 | 52.69 |
| quarot_unfused | 8 | 4096 | decode_ms_per_token | 298.3 | 298.3 | 0.338 | 298.6 | 298.6 |
| fp16_hf | 1 | 10 | prefill_ms | 39.84 | 39.61 | 0.5753 | 40.32 | 40.41 |
| fused_quarot | 1 | 10 | prefill_ms | 63.82 | 63.85 | 0.3675 | 64.11 | 64.14 |
| quarot_unfused | 1 | 10 | prefill_ms | 72.15 | 72.27 | 7.867 | 78.42 | 79.19 |
| fp16_hf | 1 | 128 | prefill_ms | 61.26 | 61.23 | 0.3975 | 61.59 | 61.63 |
| fused_quarot | 1 | 128 | prefill_ms | 77.13 | 77.29 | 0.4128 | 77.4 | 77.42 |
| quarot_unfused | 1 | 128 | prefill_ms | 80.98 | 80.5 | 3.717 | 84.03 | 84.47 |
| fp16_hf | 1 | 1024 | prefill_ms | 166.4 | 166.1 | 0.8256 | 167.1 | 167.2 |
| fused_quarot | 1 | 1024 | prefill_ms | 225.3 | 225.3 | 0.09165 | 225.4 | 225.4 |
| quarot_unfused | 1 | 1024 | prefill_ms | 225.9 | 225.9 | 0.1301 | 226 | 226 |
| fp16_hf | 1 | 4096 | prefill_ms | 814.8 | 820 | 9.956 | 820.9 | 821 |
| fused_quarot | 1 | 4096 | prefill_ms | 892.2 | 891.5 | 1.463 | 893.4 | 893.7 |
| quarot_unfused | 1 | 4096 | prefill_ms | 992.3 | 891.8 | 174.6 | 1134 | 1164 |
| fp16_hf | 2 | 10 | prefill_ms | 43 | 42.01 | 2.542 | 45.11 | 45.5 |
| fused_quarot | 2 | 10 | prefill_ms | 68.81 | 68.71 | 3.439 | 71.58 | 71.94 |
| quarot_unfused | 2 | 10 | prefill_ms | 69.07 | 69.29 | 0.5572 | 69.44 | 69.46 |
| fp16_hf | 2 | 128 | prefill_ms | 67.74 | 67.73 | 0.04179 | 67.78 | 67.78 |
| fused_quarot | 2 | 128 | prefill_ms | 90.91 | 90.77 | 0.3062 | 91.17 | 91.22 |
| quarot_unfused | 2 | 128 | prefill_ms | 90.46 | 90.34 | 0.4008 | 90.79 | 90.85 |
| fp16_hf | 2 | 1024 | prefill_ms | 326.3 | 326.9 | 1.431 | 327.2 | 327.3 |
| fused_quarot | 2 | 1024 | prefill_ms | 381.4 | 381.2 | 0.4369 | 381.8 | 381.8 |
| quarot_unfused | 2 | 1024 | prefill_ms | 381.5 | 381.3 | 0.3299 | 381.7 | 381.8 |
| fp16_hf | 2 | 4096 | prefill_ms | 1603 | 1602 | 3.559 | 1606 | 1607 |
| fused_quarot | 2 | 4096 | prefill_ms | 1741 | 1742 | 2.516 | 1742 | 1742 |
| quarot_unfused | 2 | 4096 | prefill_ms | 1740 | 1739 | 3.208 | 1743 | 1743 |
| fp16_hf | 4 | 10 | prefill_ms | 43.58 | 43.58 | 0.1603 | 43.71 | 43.72 |
| fused_quarot | 4 | 10 | prefill_ms | 73.69 | 70.88 | 5.727 | 78.4 | 79.34 |
| quarot_unfused | 4 | 10 | prefill_ms | 72.79 | 72.31 | 3.012 | 75.27 | 75.64 |
| fp16_hf | 4 | 128 | prefill_ms | 97.5 | 97.54 | 0.1241 | 97.59 | 97.6 |
| fused_quarot | 4 | 128 | prefill_ms | 125.7 | 125.7 | 0.1115 | 125.7 | 125.7 |
| quarot_unfused | 4 | 128 | prefill_ms | 125.3 | 125.3 | 0.08323 | 125.3 | 125.3 |
| fp16_hf | 4 | 1024 | prefill_ms | 689.9 | 690 | 0.5088 | 690.3 | 690.4 |
| fused_quarot | 4 | 1024 | prefill_ms | 761.8 | 762 | 0.3571 | 762.1 | 762.1 |
| quarot_unfused | 4 | 1024 | prefill_ms | 761.1 | 760.9 | 0.506 | 761.5 | 761.6 |
| fp16_hf | 4 | 4096 | prefill_ms | 3203 | 3204 | 3.134 | 3205 | 3206 |
| fused_quarot | 4 | 4096 | prefill_ms | 3551 | 3551 | 1.072 | 3551 | 3552 |
| quarot_unfused | 4 | 4096 | prefill_ms | 3548 | 3549 | 2.478 | 3550 | 3550 |
| fp16_hf | 8 | 10 | prefill_ms | 51.35 | 51.36 | 0.1578 | 51.48 | 51.49 |
| fused_quarot | 8 | 10 | prefill_ms | 74.37 | 71.74 | 5.137 | 78.58 | 79.43 |
| quarot_unfused | 8 | 10 | prefill_ms | 85.23 | 85.4 | 1.762 | 86.6 | 86.75 |
| fp16_hf | 8 | 128 | prefill_ms | 159 | 159 | 0.1233 | 159.1 | 159.1 |
| fused_quarot | 8 | 128 | prefill_ms | 219.5 | 219.5 | 0.2213 | 219.7 | 219.7 |
| quarot_unfused | 8 | 128 | prefill_ms | 219 | 219.1 | 0.399 | 219.3 | 219.3 |
| fp16_hf | 8 | 1024 | prefill_ms | 1336 | 1336 | 0.235 | 1336 | 1336 |
| fused_quarot | 8 | 1024 | prefill_ms | 1471 | 1471 | 0.8006 | 1472 | 1472 |
| quarot_unfused | 8 | 1024 | prefill_ms | 1469 | 1469 | 0.4692 | 1470 | 1470 |
| fp16_hf | 8 | 4096 | prefill_ms | 6367 | 6367 | 1.249 | 6368 | 6368 |
| fused_quarot | 8 | 4096 | prefill_ms | 7289 | 7289 | 1.057 | 7290 | 7290 |
| quarot_unfused | 8 | 4096 | prefill_ms | 7286 | 7286 | 0.4781 | 7287 | 7287 |

## Logits / Generation Quality

| reference_mode | candidate_mode | prompt_id | max_length | max_error | mean_error | top1_match | top10_overlap | kl_ref_to_out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16_hf | quarot_unfused | 0 | 16 | 22.4 | 2.131 | 0 | 0 | 12.37 |
| fp16_hf | quarot_unfused | 0 | 128 | 22.4 | 2.131 | 0 | 0 | 12.37 |
| fp16_hf | quarot_unfused | 0 | 1024 | 22.4 | 2.131 | 0 | 0 | 12.37 |
| fp16_hf | quarot_unfused | 1 | 16 | 13.36 | 2.089 | 0 | 0 | 4.211 |
| fp16_hf | quarot_unfused | 1 | 128 | 13.36 | 2.089 | 0 | 0 | 4.211 |
| fp16_hf | quarot_unfused | 1 | 1024 | 13.36 | 2.089 | 0 | 0 | 4.211 |
| fp16_hf | quarot_unfused | 2 | 16 | 13.21 | 2.029 | 0 | 0 | 4.939 |
| fp16_hf | quarot_unfused | 2 | 128 | 13.21 | 2.029 | 0 | 0 | 4.939 |
| fp16_hf | quarot_unfused | 2 | 1024 | 13.21 | 2.029 | 0 | 0 | 4.939 |
| fp16_hf | fused_quarot | 0 | 16 | 22.55 | 2.15 | 0 | 0 | 12.58 |
| fp16_hf | fused_quarot | 0 | 128 | 22.55 | 2.15 | 0 | 0 | 12.58 |
| fp16_hf | fused_quarot | 0 | 1024 | 22.55 | 2.15 | 0 | 0 | 12.58 |
| fp16_hf | fused_quarot | 1 | 16 | 13.43 | 2.077 | 0 | 0 | 4.127 |
| fp16_hf | fused_quarot | 1 | 128 | 13.43 | 2.077 | 0 | 0 | 4.127 |
| fp16_hf | fused_quarot | 1 | 1024 | 13.43 | 2.077 | 0 | 0 | 4.127 |
| fp16_hf | fused_quarot | 2 | 16 | 13.06 | 2.026 | 0 | 0 | 5.036 |
| fp16_hf | fused_quarot | 2 | 128 | 13.06 | 2.026 | 0 | 0 | 5.036 |
| fp16_hf | fused_quarot | 2 | 1024 | 13.06 | 2.026 | 0 | 0 | 5.036 |
| quarot_unfused | fused_quarot | 0 | 16 | 1.767 | 0.2762 | 1 | 0.9 | 0.03366 |
| quarot_unfused | fused_quarot | 0 | 128 | 1.767 | 0.2762 | 1 | 0.9 | 0.03366 |
| quarot_unfused | fused_quarot | 0 | 1024 | 1.767 | 0.2762 | 1 | 0.9 | 0.03366 |
| quarot_unfused | fused_quarot | 1 | 16 | 1.725 | 0.2769 | 1 | 0.8 | 0.05168 |
| quarot_unfused | fused_quarot | 1 | 128 | 1.725 | 0.2769 | 1 | 0.8 | 0.05168 |
| quarot_unfused | fused_quarot | 1 | 1024 | 1.725 | 0.2769 | 1 | 0.8 | 0.05168 |
| quarot_unfused | fused_quarot | 2 | 16 | 1.773 | 0.2543 | 0 | 0.8 | 0.03511 |
| quarot_unfused | fused_quarot | 2 | 128 | 1.773 | 0.2543 | 0 | 0.8 | 0.03511 |
| quarot_unfused | fused_quarot | 2 | 1024 | 1.773 | 0.2543 | 0 | 0.8 | 0.03511 |

Generation samples:

- `llama31_formal_quality_fp16_vs_unfused/generation_samples.md`
- `llama31_formal_quality_fp16_vs_fused/generation_samples.md`
- `llama31_formal_quality_unfused_vs_fused/generation_samples.md`

## rocprofv3 Summary

# Llama-3.1 Full Model rocprof Summary

`iters=2`; totals divide rocprof kernel stats by iterations.

| workload | kernel_calls_per_iter | kernel_time_us_per_iter |
| --- | --- | --- |
| fp16_hf_B1_L128 | 4660.500 | 173116.639 |
| fp16_hf_B1_L4096 | 4660.000 | 1703448.907 |
| fp16_hf_B4_L1024 | 4660.000 | 1429737.535 |
| fused_quarot_B1_L128 | 14340.500 | 214751.637 |
| fused_quarot_B1_L4096 | 22276.000 | 1834310.929 |
| fused_quarot_B4_L1024 | 22276.000 | 1564346.307 |
| quarot_unfused_B1_L128 | 27588.500 | 277039.320 |
| quarot_unfused_B1_L4096 | 35524.000 | 1918421.247 |
| quarot_unfused_B4_L1024 | 35524.000 | 1643087.425 |

## Top kernels: fp16_hf_B1_L128

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 1127 | 279148.636 | 247.692 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 1156 | 9865.084 | 8.534 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 256 | 9004.636 | 35.174 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 192 | 5321.301 | 27.715 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 704 | 4746.956 | 6.743 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 640 | 4663.253 | 7.286 |
| vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 469 | 4293.636 | 9.155 |
| vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda | 455 | 4240.427 | 9.320 |
| reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int, float, 4, 4 | 455 | 3966.240 | 8.717 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 455 | 3878.547 | 8.524 |

## Top kernels: fp16_hf_B1_L4096

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 871 | 1529200.890 | 1755.684 |
| attn_fwd | 224 | 843716.910 | 3766.593 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 378281.808 | 1477.663 |
| vectorized_elementwise_kernel<8, BinaryFunctor<c10::Half, c10::Half, c10::Half, binary_internal::Mul | 419 | 73693.306 | 175.879 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 73622.310 | 287.587 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 704 | 68949.453 | 97.940 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 1156 | 61598.918 | 53.286 |
| vectorized_elementwise_kernel<4, (anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>( | 455 | 61121.356 | 134.333 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 455 | 53707.340 | 118.038 |
| vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 469 | 49697.875 | 105.966 |

## Top kernels: fp16_hf_B4_L1024

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 871 | 1540982.122 | 1769.210 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 381910.140 | 1491.836 |
| attn_fwd | 224 | 262425.618 | 1171.543 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 75435.671 | 294.671 |
| vectorized_elementwise_kernel<8, BinaryFunctor<c10::Half, c10::Half, c10::Half, binary_internal::Mul | 224 | 73325.900 | 327.348 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 704 | 70762.722 | 100.515 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 1351 | 70254.531 | 52.002 |
| vectorized_elementwise_kernel<4, (anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>( | 455 | 62841.396 | 138.113 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 455 | 54909.809 | 120.681 |
| vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 469 | 49604.782 | 105.767 |

## Top kernels: fused_quarot_B1_L128

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 1127 | 283739.118 | 251.765 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 7680 | 40086.517 | 5.220 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 4224 | 13929.366 | 3.298 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 256 | 10609.622 | 41.444 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 1156 | 10033.006 | 8.679 |
| vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda | 1159 | 6085.899 | 5.251 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 192 | 5354.283 | 27.887 |
| vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 917 | 5161.555 | 5.629 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 704 | 4768.558 | 6.774 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 647 | 4695.324 | 7.257 |

## Top kernels: fused_quarot_B1_L4096

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 871 | 1517166.350 | 1741.867 |
| attn_fwd | 128 | 821122.207 | 6415.017 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 377309.290 | 1473.864 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 15616 | 124327.446 | 7.962 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 4224 | 76848.870 | 18.193 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 73799.107 | 288.278 |
| vectorized_elementwise_kernel<8, BinaryFunctor<c10::Half, c10::Half, c10::Half, binary_internal::Mul | 323 | 73592.123 | 227.839 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 704 | 68719.566 | 97.613 |
| vectorized_elementwise_kernel<4, (anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>( | 455 | 62716.435 | 137.838 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 1156 | 61146.312 | 52.895 |

## Top kernels: fused_quarot_B4_L1024

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 871 | 1536321.207 | 1763.859 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 382180.243 | 1492.892 |
| attn_fwd | 128 | 253105.299 | 1977.385 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 15616 | 127980.638 | 8.195 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 4224 | 76721.686 | 18.163 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 74181.032 | 289.770 |
| vectorized_elementwise_kernel<8, BinaryFunctor<c10::Half, c10::Half, c10::Half, binary_internal::Mul | 128 | 73238.396 | 572.175 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 704 | 70202.430 | 99.719 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 1351 | 70176.396 | 51.944 |
| vectorized_elementwise_kernel<4, (anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>( | 455 | 63116.996 | 138.719 |

## Top kernels: quarot_unfused_B1_L128

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 1127 | 288028.851 | 255.571 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 16128 | 86234.251 | 5.347 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 8448 | 30289.180 | 3.585 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 3648 | 16661.708 | 4.567 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 1156 | 10111.784 | 8.747 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 256 | 9145.684 | 35.725 |
| vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda | 1831 | 7744.618 | 4.230 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl<direct_copy_kernel_cuda(at::TensorIteratorB | 1216 | 7653.300 | 6.294 |
| vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 1397 | 7392.911 | 5.292 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 192 | 5474.272 | 28.512 |

## Top kernels: quarot_unfused_B1_L4096

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 871 | 1531010.061 | 1757.761 |
| attn_fwd | 128 | 819210.959 | 6400.086 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 377531.799 | 1474.734 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 24064 | 163818.311 | 6.808 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 8448 | 90411.249 | 10.702 |
| vectorized_elementwise_kernel<8, BinaryFunctor<c10::Half, c10::Half, c10::Half, binary_internal::Mul | 323 | 73625.587 | 227.943 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 73088.460 | 285.502 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 704 | 68226.001 | 96.912 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 743 | 66558.799 | 89.581 |
| vectorized_elementwise_kernel<4, (anonymous namespace)::pow_tensor_scalar_kernel_impl<float, float>( | 455 | 61166.140 | 134.431 |

## Top kernels: quarot_unfused_B4_L1024

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 871 | 1543043.823 | 1771.577 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB1_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 380902.175 | 1487.899 |
| attn_fwd | 128 | 252409.889 | 1971.952 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 24064 | 161294.720 | 6.703 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 8448 | 88152.695 | 10.435 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 256 | 74561.387 | 291.255 |
| vectorized_elementwise_kernel<8, BinaryFunctor<c10::Half, c10::Half, c10::Half, binary_internal::Mul | 128 | 73272.670 | 572.443 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 1351 | 70458.695 | 52.153 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 704 | 69846.230 | 99.213 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 743 | 68332.192 | 91.968 |


## 結論

- Formal wrapper 已讓 `quarot_unfused` / `fused_quarot` 透過正式 token-by-token API 執行 `prefill()`、`decode_one()`、`generate()`。
- Fused QuaRot 相對 QuaRot unfused 的 decode latency 有穩定改善；在完整 8B path 中主要落在約 `2.3x-5.7x`，shape-dependent。
- Fused QuaRot 相對 FP16 HF decode 多數 shape 接近或略慢，因為 GEMM/projection/prefill 仍主導 full-model 成本；不可宣稱 end-to-end 100x speedup。
- QuaRot path 尚未完成正式權重旋轉/calibration，FP16 vs QuaRot 的 logits/generation 品質目前不代表最終模型品質。
