# Single Decoder Layer / QuaRot Fusion 評估報告

## 實驗環境

- 日期：2026-07-04
- 容器：`rocm/vllm-dev:rocm7.2_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.14.0rc0`
- GPU/ROCm：使用本機 ROCm HIP runtime；rocprof-compute 目前不支援 `gfx1201`，rocprofv3 可收 kernel trace/stats。
- 程式：`single_decoder_layer_benchmark.py`，輸出在 `single_decoder_layer_results/`，rocprof 在 `rocprof_single_decoder_layer/`。

## Benchmark 設定

- Decoder layer decode-step harness：前 `L-1` token KV cache 已 prefilled；timed region 包含 current token RMSNorm、QKV projection、RoPE/KV append、attention decode、O projection/residual、FFN gate/up/down projection。
- 三條主路徑：`fp16_baseline`、`quarot_unfused`、`fused_quarot`。
- Ablation：`quarot_unfused`、`k1_fused`、`k1_k2_fused`、`attention_fused`、`fused_quarot`。
- Shapes：batch = 1,2,4,8；context length L = 10,128,1024,4096；FFN hidden = 11008,14336。
- Timing：HIP/PyTorch CUDA event，`warmup=2`、`iters=5`。
- 權重/輸入：固定 random seed；三條路徑共用同一組 current hidden、cache state 與 layer weights。

## Correctness

下表為 fused QuaRot output hidden states 相對 QuaRot original unfused 的差異。relative error 會被接近 0 的元素放大，因此判讀時以 max/mean error 為主。

| batch | seq_len | ffn_hidden | max_error | mean_error | mean_relative_error | tolerance | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 14336 | 1.168 | 0.232 | 0.677 | max<=1.25,mean<=0.25 | PASS |
| 1 | 128 | 14336 | 0.615 | 0.123 | 0.518 | max<=1.25,mean<=0.25 | PASS |
| 1 | 1024 | 14336 | 0.180 | 0.038 | 11.897 | max<=1.25,mean<=0.25 | PASS |
| 1 | 4096 | 14336 | 0.119 | 0.026 | 13.297 | max<=1.25,mean<=0.25 | PASS |

完整 correctness CSV：`single_decoder_layer_results/correctness.csv`。

## Latency

全 shape 平均：fused QuaRot 相對 QuaRot unfused speedup = 6.36x，範圍 3.81x 到 21.16x；相對 FP16 baseline 平均 = 1.21x，範圍 0.74x 到 2.29x。

代表表：batch=1、FFN hidden=14336。

| batch | seq_len | ffn_hidden | fp16_ms | unfused_ms | fused_ms | fused_vs_unfused | fused_vs_fp16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 14336 | 1.174 | 4.348 | 1.077 | 4.038 | 1.090 |
| 1 | 128 | 14336 | 1.149 | 4.242 | 1.082 | 3.919 | 1.062 |
| 1 | 1024 | 14336 | 1.235 | 4.641 | 1.126 | 4.123 | 1.097 |
| 1 | 4096 | 14336 | 1.530 | 5.639 | 1.300 | 4.339 | 1.177 |

完整 latency CSV：`single_decoder_layer_results/latency.csv`。

## Ablation

代表表：batch=1、FFN hidden=14336。

| seq_len | variant | latency_ms | speedup_vs_unfused |
| --- | --- | --- | --- |
| 10 | quarot_unfused | 5.205 | 1.000 |
| 10 | k1_fused | 2.825 | 1.842 |
| 10 | k1_k2_fused | 2.353 | 2.212 |
| 10 | attention_fused | 1.639 | 3.176 |
| 10 | fused_quarot | 1.140 | 4.566 |
| 128 | quarot_unfused | 5.538 | 1.000 |
| 128 | k1_fused | 5.543 | 0.999 |
| 128 | k1_k2_fused | 2.783 | 1.990 |
| 128 | attention_fused | 1.639 | 3.379 |
| 128 | fused_quarot | 1.080 | 5.126 |
| 1024 | quarot_unfused | 5.145 | 1.000 |
| 1024 | k1_fused | 2.523 | 2.039 |
| 1024 | k1_k2_fused | 2.523 | 2.040 |
| 1024 | attention_fused | 1.629 | 3.157 |
| 1024 | fused_quarot | 1.130 | 4.552 |
| 4096 | quarot_unfused | 5.796 | 1.000 |
| 4096 | k1_fused | 4.871 | 1.190 |
| 4096 | k1_k2_fused | 2.465 | 2.351 |
| 4096 | attention_fused | 1.775 | 3.265 |
| 4096 | fused_quarot | 1.292 | 4.487 |

完整 ablation CSV：`single_decoder_layer_results/ablation.csv`。

## rocprof Kernel Breakdown

rocprofv3 代表點：batch=1、L=128、FFN hidden=14336、iters=5。
rocprofv3 執行期間出現 timestamp swap warnings；CSV stats 已產生，但細粒度 kernel duration 仍應視為 profiling 近似值。

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

## Memory / Counter 補充

gfx1201 上 GL2C/LDS/occupancy counter 先前量測仍為 0；rocprof-compute 也回報不支援此 arch。因此本報告用 rocprofv3 kernel time/calls 搭配理論最小 memory traffic 補充。

| component | formula | bytes |
| --- | --- | --- |
| K2 FP16 KV cache read | B * L * D * 2(K,V) * 2 bytes | 2097152 |
| K2 INT4 KV cache read | B * L * D * 2(K,V) * 0.5 bytes + scales | 557056 |
| K1 current read/write | read FP16 K/V, write INT4 K/V + scale | 20736 |
| K3 output quant | read FP16 4096, write INT4 4096 + 16 scales | 10272 |
| FFN intermediate quant | read FP16 gate/up, write INT4 hidden + scales | 64624 |

以上 bytes 為代表點 B=1、L=128、FFN hidden=14336 的最低資料量估算，未包含 GEMM 讀權重、cache miss、alignment、temporary tensor、PyTorch dispatcher overhead。

## 結論與限制

- 這次結果比原本只測 PyTorch reference pipeline 的 kernel-fusion ablation 更可信，因為 harness 包含 norm、projection、residual、attention decode、O/FFN projection，且只把 K1/K2/K3/FFN 替換成 fused kernel。
- Fused QuaRot 對 QuaRot original unfused path 有穩定 decoder-layer speedup，主要來自減少 separate Hadamard/quant/dequant 的 PyTorch kernel calls。
- 相對 FP16 baseline 的改善有限且 shape-dependent，因為完整 decoder layer 中 GEMM/projection 仍佔主要時間；因此不能宣稱 full LLM end-to-end 有 100x speedup。
- 目前仍是 synthetic single decoder layer harness，使用 random weights/cache；不是完整 HuggingFace/Llama model class，也未量測 tokenizer、sampling、多層模型排程與真實 logits quality。
- correctness 目前固定輸出 max/mean/relative error；fused K1 pack 與 unfused reference 允許少量 INT4 byte mismatch，最終 hidden mean error 需依專題可接受 tolerance 再定案。
