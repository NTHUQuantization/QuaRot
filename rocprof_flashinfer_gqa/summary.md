# Single Decoder Layer rocprof Summary

`iters=20`; times below divide rocprof kernel totals by iterations.

| variant | batch | seq_len | ffn_hidden | kernel_calls_per_iter | kernel_time_us_per_iter |
| --- | --- | --- | --- | --- | --- |
| gqa_B1_L4096 |  |  |  | 38.800 | 2253.865 |

## Top kernels: gqa_B1_L4096

| kernel | calls | total_us | avg_us |
| --- | --- | --- | --- |
| void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::RotaryMode)0, false, 8ul, 16ul,  | 26 | 8746.873 | 336.418 |
| void flashinfer::BatchDecodeWithPagedKVGQAKernel<(flashinfer::RotaryMode)0, false, 8ul, 16ul, 8u | 26 | 8653.726 | 332.836 |
| void elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at: | 370 | 8157.305 | 22.047 |
| void flashinfer::BatchDecodeWithPagedKVGQAKernel<(flashinfer::RotaryMode)0, false, 16ul, 8ul, 16 | 26 | 5261.911 | 202.381 |
| void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::RotaryMode)0, false, 16ul, 8ul,  | 26 | 5237.041 | 201.425 |
| void vectorized_elementwise_kernel<8, CUDAFunctor_add<c10::Half>, array<char*, 3ul> >(int, CUDAF | 56 | 1922.816 | 34.336 |
| void vectorized_elementwise_kernel<4, float16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&) | 14 | 1249.854 | 89.275 |
| void vectorized_elementwise_kernel<4, AbsFunctor<float>, array<char*, 2ul> >(int, AbsFunctor<flo | 8 | 816.283 | 102.035 |
| void elementwise_kernel_manual_unroll<128, 8, gpu_kernel_impl_nocast<direct_copy_kernel_cuda(at: | 128 | 811.837 | 6.342 |
| void reduce_kernel<512, 1, ReduceOp<float, func_wrapper_t<float, MaxNanFunctor<float> >, unsigne | 6 | 623.994 | 103.999 |
| void elementwise_kernel_manual_unroll<128, 4, gpu_kernel_impl_nocast<BinaryFunctor<float, float, | 4 | 568.035 | 142.009 |
| void vectorized_elementwise_kernel<4, (anonymous namespace)::launch_clamp_scalar(at::TensorItera | 10 | 543.084 | 54.308 |
