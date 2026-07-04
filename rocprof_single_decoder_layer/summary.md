# Single Decoder Layer rocprof Summary

`iters=5`; times below divide rocprof kernel totals by iterations.

| variant | batch | seq_len | ffn_hidden | kernel_calls_per_iter | kernel_time_us_per_iter |
| --- | --- | --- | --- | --- | --- |
| fp16_baseline | 1 | 128 | 14336 | 97.800 | 1586.151 |
| fused_quarot | 1 | 128 | 14336 | 94.800 | 1599.432 |
| quarot_unfused | 1 | 128 | 14336 | 390.800 | 2582.754 |

## Top kernels: fp16_baseline

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_AS | 40 | 4940.720 | 123.518 |
| void vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAF | 43 | 441.560 | 10.269 |
| void elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at: | 60 | 367.703 | 6.128 |
| void elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, | 45 | 244.338 | 5.430 |
| __amd_rocclr_fillBufferAligned | 1 | 202.105 | 202.105 |
| void vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&) | 41 | 194.190 | 4.736 |
| void vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda | 24 | 172.509 | 7.188 |
| void vectorized_elementwise_kernel<4, CUDAFunctor_add<float>, array<char*, 3ul> >(int, CUDAFunct | 14 | 158.071 | 11.291 |
| void elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at: | 14 | 136.099 | 9.721 |
| void reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float> >, unsigne | 2 | 86.297 | 43.148 |

## Top kernels: fused_quarot

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_AS | 35 | 4776.105 | 136.460 |
| void vectorized_elementwise_kernel<4, float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda | 24 | 445.384 | 18.558 |
| void vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAF | 38 | 322.248 | 8.480 |
| void elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at: | 60 | 247.830 | 4.130 |
| void elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, | 30 | 247.395 | 8.246 |
| void vectorized_elementwise_kernel<8, CUDAFunctorOnSelf_add<short>, array<char*, 2ul> >(int, CUD | 22 | 240.979 | 10.954 |
| void vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&) | 36 | 190.662 | 5.296 |
| void elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at: | 24 | 176.527 | 7.355 |
| __amd_rocclr_fillBufferAligned | 1 | 166.154 | 166.154 |
| void vectorized_elementwise_kernel<4, CUDAFunctor_add<float>, array<char*, 3ul> >(int, CUDAFunct | 4 | 140.856 | 35.214 |

## Top kernels: quarot_unfused

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC1_AFEM1_AFEM1_AS | 35 | 4743.955 | 135.542 |
| void elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at: | 500 | 2331.689 | 4.663 |
| void elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at: | 204 | 897.197 | 4.398 |
| void vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAF | 258 | 705.709 | 2.735 |
| void elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl<direct_copy_kernel_cuda(at::Tensor | 52 | 354.077 | 6.809 |
| void elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, | 60 | 334.479 | 5.575 |
| void vectorized_elementwise_kernel<4, CUDAFunctor_add<float>, array<char*, 3ul> >(int, CUDAFunct | 94 | 334.259 | 3.556 |
| void vectorized_elementwise_kernel<8, CUDAFunctorOnSelf_add<short>, array<char*, 2ul> >(int, CUD | 52 | 317.975 | 6.115 |
| void vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&) | 76 | 283.011 | 3.724 |
| void elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl<direct_copy_kernel_cuda(at::Tensor | 37 | 261.686 | 7.073 |
