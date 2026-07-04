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
