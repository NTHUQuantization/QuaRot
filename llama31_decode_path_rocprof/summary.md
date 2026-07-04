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
