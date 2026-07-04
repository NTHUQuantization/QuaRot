# Llama-3.1 Full Model rocprof Summary

`iters=5`; totals divide rocprof kernel stats by iterations.

| workload | kernel_calls_per_iter | kernel_time_us_per_iter |
| --- | --- | --- |
| fp16_hf_B1_L128 | 4681.600 | 154454.571 |

## Top kernels: fp16_hf_B1_L128

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_ | 2639 | 598417.221 | 226.759 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<BinaryFunctor<c10::Half, c10::Half,  | 2440 | 20039.505 | 8.213 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x64x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 512 | 17904.077 | 34.969 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at::TensorIt | 1216 | 13931.393 | 11.457 |
| vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1 | 1485 | 13083.836 | 8.811 |
| Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_ASEM1_CL | 448 | 12446.253 | 27.782 |
| vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAFunctor_ad | 1408 | 10352.915 | 7.353 |
| elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, float, b | 975 | 9817.103 | 10.069 |
| (anonymous namespace)::CatArrayBatchedCopy<(anonymous namespace)::OpaqueType<2u>, unsigned int, 4, 6 | 1472 | 9780.034 | 6.644 |
| elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<CUDAFunctor_add<c10::Half> >(at::Ten | 992 | 9450.834 | 9.527 |
