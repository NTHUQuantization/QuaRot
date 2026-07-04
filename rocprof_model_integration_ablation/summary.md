| Variant | rocprof kernel time / iter (us) | Kernel calls / iter | Speedup vs unfused |
| --- | ---: | ---: | ---: |
| unfused_all | 1154.813 | 279.30 | 1.00x |
| k1_fused | 669.299 | 147.30 | 1.73x |
| k1_k2_fused | 410.052 | 133.30 | 2.82x |
| attention_fused | 278.410 | 70.30 | 4.15x |
| full_fused | 23.399 | 4.30 | 49.35x |

### unfused_all
| Top kernel group | Time / iter (us) | Calls / iter |
| --- | ---: | ---: |
| elementwise_kernel_manual_unroll | 389.813 | 88.00 |
| elementwise_kernel_manual_unroll | 155.542 | 34.00 |
| vectorized_elementwise_kernel | 125.459 | 44.00 |
| elementwise_kernel_manual_unroll | 54.547 | 6.00 |
| vectorized_elementwise_kernel | 46.037 | 16.00 |
| vectorized_elementwise_kernel | 45.701 | 6.00 |

### k1_fused
| Top kernel group | Time / iter (us) | Calls / iter |
| --- | ---: | ---: |
| elementwise_kernel_manual_unroll | 163.053 | 34.00 |
| elementwise_kernel_manual_unroll | 158.702 | 32.00 |
| elementwise_kernel_manual_unroll | 47.916 | 4.00 |
| vectorized_elementwise_kernel | 39.369 | 16.00 |
| elementwise_kernel_manual_unroll | 38.111 | 2.00 |
| vectorized_elementwise_kernel | 37.969 | 4.00 |

### k1_k2_fused
| Top kernel group | Time / iter (us) | Calls / iter |
| --- | ---: | ---: |
| elementwise_kernel_manual_unroll | 119.489 | 32.00 |
| elementwise_kernel_manual_unroll | 107.755 | 32.00 |
| vectorized_elementwise_kernel | 38.150 | 16.00 |
| vectorized_elementwise_kernel | 37.631 | 16.00 |
| elementwise_kernel_manual_unroll | 8.022 | 2.00 |
| reduce_kernel | 7.944 | 2.00 |

### attention_fused
| Top kernel group | Time / iter (us) | Calls / iter |
| --- | ---: | ---: |
| elementwise_kernel_manual_unroll | 153.103 | 32.00 |
| vectorized_elementwise_kernel | 46.055 | 16.00 |
| vectorized_elementwise_kernel | 8.234 | 2.00 |
| elementwise_kernel_manual_unroll | 5.482 | 1.00 |
| append_kv_had_quant_kernel | 5.457 | 1.00 |
| vectorized_elementwise_kernel | 5.079 | 1.00 |

### full_fused
| Top kernel group | Time / iter (us) | Calls / iter |
| --- | ---: | ---: |
| BatchDecodeWithPagedKVCacheKernel | 7.450 | 1.00 |
| append_kv_had_quant_kernel | 6.941 | 1.00 |
| output_had_quant_kernel | 4.238 | 1.00 |
| fused_ffn_silu_hadamard_quant_kernel | 3.167 | 1.00 |
| __amd_rocclr_copyBuffer | 1.009 | 0.15 |
| void (anonymous namespace)::elementwise_kernel_with_index<int, a | 0.387 | 0.10 |

