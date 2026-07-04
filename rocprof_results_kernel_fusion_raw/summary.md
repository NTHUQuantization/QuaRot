| Block | Variant | Kernel time / iter (us) | Kernel calls / iter | Global read FetchSize (KiB) | Global write est. (KiB) | Min semantic IO (KiB) | LDS inst / iter | LDS bank conflict / iter | SQ waves / iter | GPU active % | Mean occ / CU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FFN | unfused PyTorch | 220.875 | 67.00 | 0 | 0 | n/a | 0 | 0 | 4120.000 | 100.000 | 0 |
| FFN | hadacore pipeline | 140.596 | 22.00 | 0 | 0 | n/a | 0 | 0 | 1656.000 | 100.000 | 0 |
| FFN | fused prototype | 3.490 | 1.00 | 0 | 0 | 63.109 | 0 | 0 | 448.000 | 100.000 | 0 |
| K1 | unfused PyTorch, RoPE off | 437.843 | 134.55 | 0 | 0 | n/a | 0 | 0 | 918.200 | 100.000 | 0 |
| K1 | unfused PyTorch, RoPE on | 545.682 | 151.55 | 0 | 0 | n/a | 0 | 0 | 1038.200 | 100.000 | 0 |
| K1 | fused append, RoPE off | 5.331 | 1.00 | 0 | 0 | 5.062 | 0 | 0 | 32.000 | 100.000 | 0 |
| K1 | fused append, RoPE on | 9.603 | 1.00 | 0 | 0 | 5.062 | 0 | 0 | 32.000 | 100.000 | 0 |
| K2 | baseline FP16 KV decode | 383.278 | 1.00 | 0 | 0 | 16384.000 | 0 | 0 | 32.000 | 100.000 | 0 |
| K2 | QuaRot original unfused INT4 dequant + FP16 decode | 988.947 | 23.90 | 0 | 0 | n/a | 0 | 0 | 378812.200 | 100.000 | 0 |
| K2 | optimized INT4 KV decode | 234.357 | 1.00 | 0 | 0 | 4352.000 | 0 | 0 | 32.000 | 100.000 | 0 |
| K3 | unfused PyTorch | 251.394 | 64.05 | 0 | 0 | n/a | 0 | 0 | 696.800 | 100.000 | 0 |
| K3 | fused output quant | 3.412 | 1.00 | 0 | 0 | 10.031 | 0 | 0 | 128.000 | 100.000 | 0 |

Notes: FetchSize uses rocprofv3 FetchSize when present, otherwise GL2C_EA_RDREQ_{32,64,128}B_sum. Write bytes are estimated as GL2C_EA_WRREQ_64B_sum * 64. Min semantic IO is an analytical lower bound for fused/custom kernels and K2 cache reads; PyTorch unfused pipelines have additional intermediate traffic. Time is summed across selected kernels and divided by 20 iterations.
