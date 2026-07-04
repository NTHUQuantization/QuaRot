# Llama-3.1 8B Decode Path QuaRot Fusion 評估

## 範圍

- 使用 `meta-llama/Llama-3.1-8B` 真實 HF 權重與 tokenizer。
- Prefill 由 HF model 產生每層 prompt KV cache；decode step 手動走 32 層 decoder。
- `fp16_manual` 使用相同 projection/norm/residual/MLP，attention decode 走 GQA-aware FP16 paged KV kernel，並以 HF decode logits 驗證接線。
- `quarot_unfused` 使用 PyTorch Hadamard/INT4 pack/dequant/reference steps；`fused_quarot` 使用目前 HIP K1/K2/K3/FFN fused kernels。
- 這是 full model decoder-step path，不是 transformers `generate()` 端到端 patch。

## Latency Summary

| variant | batch | context_len | mean | median | std | p90 | p95 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16_manual | 1 | 10 | 37.82 | 37.82 | 0.03172 | 37.85 | 37.85 | 37.79 | 37.86 |
| fp16_manual | 1 | 128 | 38.65 | 38.24 | 0.8168 | 39.32 | 39.45 | 38.12 | 39.59 |
| fp16_manual | 1 | 1024 | 41.17 | 41.18 | 0.0539 | 41.22 | 41.22 | 41.12 | 41.23 |
| fp16_manual | 1 | 4096 | 50.84 | 50.81 | 0.1037 | 50.93 | 50.94 | 50.75 | 50.95 |
| fp16_manual | 2 | 10 | 38.04 | 38.05 | 0.03368 | 38.06 | 38.07 | 38 | 38.07 |
| fp16_manual | 2 | 128 | 38.47 | 38.48 | 0.1552 | 38.59 | 38.61 | 38.31 | 38.62 |
| fp16_manual | 2 | 1024 | 41.3 | 41.3 | 0.04769 | 41.34 | 41.35 | 41.26 | 41.35 |
| fp16_manual | 4 | 10 | 38.73 | 38.71 | 0.03887 | 38.76 | 38.76 | 38.7 | 38.77 |
| fp16_manual | 4 | 128 | 38.94 | 38.9 | 0.08192 | 39.01 | 39.02 | 38.88 | 39.03 |
| fp16_manual | 4 | 1024 | 41.85 | 41.8 | 0.09238 | 41.93 | 41.94 | 41.8 | 41.96 |
| fused_quarot | 1 | 10 | 40.67 | 40.7 | 0.07711 | 40.73 | 40.73 | 40.58 | 40.73 |
| fused_quarot | 1 | 128 | 40.89 | 40.89 | 0.03514 | 40.92 | 40.92 | 40.86 | 40.93 |
| fused_quarot | 1 | 1024 | 43.84 | 44.19 | 0.9005 | 44.44 | 44.48 | 42.81 | 44.51 |
| fused_quarot | 1 | 4096 | 48.14 | 48.12 | 0.04044 | 48.17 | 48.18 | 48.11 | 48.18 |
| fused_quarot | 2 | 10 | 41.03 | 40.97 | 0.1061 | 41.11 | 41.13 | 40.96 | 41.15 |
| fused_quarot | 2 | 128 | 41.31 | 41.16 | 0.2671 | 41.52 | 41.57 | 41.15 | 41.61 |
| fused_quarot | 2 | 1024 | 42.81 | 42.84 | 0.05026 | 42.84 | 42.84 | 42.75 | 42.84 |
| fused_quarot | 4 | 10 | 41.47 | 41.46 | 0.0303 | 41.49 | 41.5 | 41.44 | 41.5 |
| fused_quarot | 4 | 128 | 41.59 | 41.6 | 0.01104 | 41.6 | 41.6 | 41.58 | 41.6 |
| fused_quarot | 4 | 1024 | 43.62 | 43.49 | 0.249 | 43.82 | 43.86 | 43.46 | 43.91 |
| quarot_unfused | 1 | 10 | 112.5 | 109.9 | 5.302 | 116.8 | 117.7 | 109 | 118.6 |
| quarot_unfused | 1 | 128 | 119.9 | 119.3 | 1.066 | 120.8 | 120.9 | 119.3 | 121.1 |
| quarot_unfused | 1 | 1024 | 119.2 | 117.6 | 6.178 | 124.3 | 125.2 | 114 | 126 |
| quarot_unfused | 1 | 4096 | 112.7 | 112.4 | 2.855 | 115 | 115.4 | 110 | 115.7 |
| quarot_unfused | 2 | 10 | 113.8 | 111.2 | 5.125 | 118 | 118.8 | 110.4 | 119.7 |
| quarot_unfused | 2 | 128 | 122.7 | 121.8 | 4.006 | 126 | 126.6 | 119.2 | 127.1 |
| quarot_unfused | 2 | 1024 | 116.9 | 115.9 | 1.709 | 118.2 | 118.5 | 115.8 | 118.8 |
| quarot_unfused | 4 | 10 | 123.5 | 122.6 | 5.283 | 127.9 | 128.5 | 118.7 | 129.2 |
| quarot_unfused | 4 | 128 | 116.5 | 114.9 | 5.738 | 121.3 | 122.1 | 111.7 | 122.9 |
| quarot_unfused | 4 | 1024 | 116.8 | 114.6 | 5.371 | 121.3 | 122.1 | 113 | 123 |

## Speedup Summary

| batch | context_len | fp16_manual_ms | quarot_unfused_ms | fused_quarot_ms | fused_vs_fp16 | fused_vs_quarot_unfused |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 37.82 | 112.5 | 40.67 | 0.93 | 2.766 |
| 1 | 128 | 38.65 | 119.9 | 40.89 | 0.9451 | 2.932 |
| 1 | 1024 | 41.17 | 119.2 | 43.84 | 0.9393 | 2.719 |
| 1 | 4096 | 50.84 | 112.7 | 48.14 | 1.056 | 2.341 |
| 2 | 10 | 38.04 | 113.8 | 41.03 | 0.9272 | 2.773 |
| 2 | 128 | 38.47 | 122.7 | 41.31 | 0.9313 | 2.97 |
| 2 | 1024 | 41.3 | 116.9 | 42.81 | 0.9648 | 2.73 |
| 4 | 10 | 38.73 | 123.5 | 41.47 | 0.9339 | 2.978 |
| 4 | 128 | 38.94 | 116.5 | 41.59 | 0.9362 | 2.801 |
| 4 | 1024 | 41.85 | 116.8 | 43.62 | 0.9596 | 2.679 |

## Correctness / Logits

| batch | context_len | variant | reference | max_error | mean_error | mean_relative_error | top1_match | top10_overlap | kl_ref_to_out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | fp16_manual | fp16_hf | 0.01562 | 0.002464 | 0.00636 | 1 | 1 | 3.431e-07 |
| 1 | 10 | fused_quarot | fp16_hf | 18.99 | 2.022 | 4.949 | 0 | 0 | 6.626 |
| 1 | 10 | quarot_unfused | fp16_hf | 18.87 | 2.011 | 4.976 | 0 | 0 | 6.934 |
| 1 | 10 | fused_quarot | quarot_unfused | 1.65 | 0.2615 | 1.225 | 1 | 0.8 | 0.02237 |
| 1 | 128 | fp16_manual | fp16_hf | 0.01562 | 0.003023 | 0.02154 | 1 | 1 | 1.581e-07 |
| 1 | 128 | fused_quarot | fp16_hf | 12.15 | 1.96 | 16.86 | 1 | 0.1 | 0.6893 |
| 1 | 128 | quarot_unfused | fp16_hf | 12.06 | 1.964 | 16.72 | 1 | 0.1 | 0.5224 |
| 1 | 128 | fused_quarot | quarot_unfused | 1.25 | 0.1918 | 0.6856 | 1 | 0.9 | 0.02373 |
| 1 | 1024 | fp16_manual | fp16_hf | 0.01562 | 0.002049 | 0.01343 | 1 | 1 | -9.794e-08 |
| 1 | 1024 | fused_quarot | fp16_hf | 12.23 | 2.067 | 11.21 | 1 | 0.1 | 0.385 |
| 1 | 1024 | quarot_unfused | fp16_hf | 12.27 | 2.06 | 11.11 | 1 | 0.1 | 0.3663 |
| 1 | 1024 | fused_quarot | quarot_unfused | 1.039 | 0.1423 | 1.18 | 1 | 0.9 | 0.005661 |
| 1 | 4096 | fp16_manual | fp16_hf | 0.02344 | 0.004516 | 0.02551 | 1 | 1 | -1.42e-08 |
| 1 | 4096 | fused_quarot | fp16_hf | 11.43 | 2.055 | 9.28 | 1 | 0.1 | 0.6317 |
| 1 | 4096 | quarot_unfused | fp16_hf | 11.44 | 2.047 | 8.971 | 1 | 0.1 | 0.7339 |
| 1 | 4096 | fused_quarot | quarot_unfused | 1.06 | 0.133 | 0.5471 | 1 | 0.9 | 0.009814 |
| 2 | 10 | fp16_manual | fp16_hf | 0.01562 | 0.002124 | 0.005701 | 1 | 1 | 8.588e-06 |
| 2 | 10 | fused_quarot | fp16_hf | 19.31 | 2.006 | 5.038 | 0 | 0 | 7.005 |
| 2 | 10 | quarot_unfused | fp16_hf | 18.87 | 2.007 | 4.933 | 0 | 0 | 6.785 |
| 2 | 10 | fused_quarot | quarot_unfused | 1.744 | 0.2725 | 1.155 | 1 | 0.85 | 0.05702 |
| 2 | 128 | fp16_manual | fp16_hf | 0.01562 | 0.002596 | 0.02415 | 1 | 1 | 3.582e-07 |
| 2 | 128 | fused_quarot | fp16_hf | 12.39 | 1.966 | 14.95 | 1 | 0.1 | 0.5824 |
| 2 | 128 | quarot_unfused | fp16_hf | 12.17 | 1.965 | 12.06 | 1 | 0.1 | 0.4957 |
| 2 | 128 | fused_quarot | quarot_unfused | 1.681 | 0.1928 | 0.7186 | 1 | 0.9 | 0.01495 |
| 2 | 1024 | fp16_manual | fp16_hf | 0.01562 | 0.002068 | 0.01568 | 1 | 1 | -5.864e-08 |
| 2 | 1024 | fused_quarot | fp16_hf | 12.38 | 2.075 | 12.04 | 1 | 0.1 | 0.3664 |
| 2 | 1024 | quarot_unfused | fp16_hf | 12.36 | 2.067 | 11.95 | 1 | 0.1 | 0.3751 |
| 2 | 1024 | fused_quarot | quarot_unfused | 1.039 | 0.1474 | 0.9518 | 1 | 0.9 | 0.004897 |
| 4 | 10 | fp16_manual | fp16_hf | 0.01562 | 0.0024 | 0.00765 | 1 | 1 | 2.088e-06 |
| 4 | 10 | fused_quarot | fp16_hf | 19.31 | 2.015 | 6.283 | 0 | 0 | 7.012 |
| 4 | 10 | quarot_unfused | fp16_hf | 19.17 | 2.012 | 6.199 | 0 | 0 | 7.118 |
| 4 | 10 | fused_quarot | quarot_unfused | 1.744 | 0.2675 | 1.064 | 1 | 0.875 | 0.04741 |
| 4 | 128 | fp16_manual | fp16_hf | 0.01562 | 0.002382 | 0.01227 | 1 | 1 | 3.537e-07 |
| 4 | 128 | fused_quarot | fp16_hf | 12.16 | 1.96 | 10.57 | 1 | 0.1 | 0.692 |
| 4 | 128 | quarot_unfused | fp16_hf | 12.29 | 1.958 | 10.5 | 1 | 0.1 | 0.6603 |
| 4 | 128 | fused_quarot | quarot_unfused | 1.51 | 0.1949 | 0.8686 | 1 | 0.925 | 0.0177 |
| 4 | 1024 | fp16_manual | fp16_hf | 0.01953 | 0.002198 | 0.01244 | 1 | 1 | -7.378e-08 |
| 4 | 1024 | fused_quarot | fp16_hf | 12.38 | 2.076 | 10.74 | 1 | 0.1 | 0.3544 |
| 4 | 1024 | quarot_unfused | fp16_hf | 12.44 | 2.067 | 10.54 | 1 | 0.1 | 0.3943 |
| 4 | 1024 | fused_quarot | quarot_unfused | 1 | 0.1482 | 0.5812 | 1 | 0.95 | 0.007982 |

## rocprofv3 Kernel Breakdown

# Llama-3.1 Full Model rocprof Summary

`iters=3`; totals divide rocprof kernel stats by iterations.

| workload | kernel_calls_per_iter | kernel_time_us_per_iter |
| --- | --- | --- |
| fp16_manual_B1_L128 | 5058.667 | 103693.391 |
| fused_quarot_B1_L128 | 6125.333 | 108428.658 |
| quarot_unfused_B1_L128 | 20845.333 | 173285.312 |

## Top kernels: fp16_manual_B1_L128

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 1159 | 239052.137 | 206.257 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 2144 | 11452.541 | 5.342 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 384 | 11231.320 | 29.248 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 1728 | 5726.731 | 3.314 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 961 | 4728.274 | 4.920 |
| vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 757 | 2887.599 | 3.815 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 512 | 2706.928 | 5.287 |
| reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int, float, 4, 4 | 458 | 2701.181 | 5.898 |
| flashinfer::BatchDecodeWithPagedKVGQAKernel<(flashinfer::RotaryMode)0, false, 8ul, 16ul, 8ul, 1ul, _ | 160 | 2587.100 | 16.169 |
| vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda | 748 | 2424.626 | 3.241 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 64 | 2314.217 | 36.160 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 455 | 2301.261 | 5.058 |

## Top kernels: fused_quarot_B1_L128

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 1159 | 241237.080 | 208.142 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 384 | 11694.276 | 30.454 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 2144 | 10236.975 | 4.775 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 1728 | 6254.370 | 3.619 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 961 | 5442.655 | 5.664 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 775 | 3693.450 | 4.766 |
| vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 917 | 3227.611 | 3.520 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 512 | 3182.420 | 6.216 |
| vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda | 908 | 2943.740 | 3.242 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl<direct_copy_kernel_cuda(at::TensorIteratorB | 704 | 2619.647 | 3.721 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl<direct_copy_kernel_cuda(at::TensorIteratorB | 648 | 2534.289 | 3.911 |
| reduce_kernel<512, 1, ReduceOp<float, MeanOps<float, float, float, float>, unsigned int, float, 4, 4 | 458 | 2488.301 | 5.433 |

## Top kernels: quarot_unfused_B1_L128

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 1159 | 247415.683 | 213.473 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 16224 | 81759.844 | 5.039 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 8768 | 28782.942 | 3.283 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 6080 | 27246.878 | 4.481 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 384 | 11611.056 | 30.237 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl<direct_copy_kernel_cuda(at::TensorIteratorB | 1664 | 10578.712 | 6.357 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl<direct_copy_kernel_cuda(at::TensorIteratorB | 968 | 7119.963 | 7.355 |
| vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 1717 | 7052.311 | 4.107 |
| vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda | 2028 | 6434.822 | 3.173 |
| vectorized_elementwise_kernel<4, CUDAFunctor_add<float>, array<char*, 3ul> >(int, CUDAFunctor_add<fl | 2562 | 6404.984 | 2.500 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 961 | 5125.738 | 5.334 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 935 | 4997.908 | 5.345 |



## 結論與限制

- `fp16_manual` 對 HF decode 的 top1/top10 皆一致，最大誤差約在 FP16 累積誤差範圍，表示 single-token decoder 接線可信。
- `fused_quarot` 相對 `quarot_unfused` 明顯降低 latency；本批結果主要代表 decoder-step projected path speedup。
- QuaRot path 尚未做完整權重旋轉與校正，因此不可把目前 logits quality 視為正式模型品質。
- profiling 若在 gfx1201 上 GL2C/LDS/occupancy counter 為 0，報告以 rocprof kernel trace/top kernels 與理論 memory traffic 補充。
