| Block | Variant | Kernel time / iter (us) | Kernel calls / iter | Global read FetchSize (KiB) | Global write est. (KiB) | LDS util | Occupancy % | Mean occ / CU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FFN | unfused PyTorch | 258.622 | 67.00 | 0 | 0 | 0 | 0 | 0 |
| FFN | hadacore pipeline | 122.235 | 22.00 | 0 | 0 | 0 | 0 | 0 |
| FFN | fused prototype | 4.469 | 1.00 | 0 | 0 | 0 | 0 | 0 |
| K1 | fused append, RoPE off | 5.392 | 1.00 | 0 | 0 | 0 | 0 | 0 |
| K1 | fused append, RoPE on | 5.174 | 1.00 | 0 | 0 | 0 | 0 | 0 |
| K2 | FP16 KV decode | 379.384 | 1.00 | 0 | 0 | 0 | 0 | 0 |
| K2 | INT4 KV decode | 235.922 | 1.00 | 0 | 0 | 0 | 0 | 0 |
| K3 | fused output quant | 3.660 | 1.00 | 0 | 0 | 0 | 0 | 0 |

Notes: FetchSize is reported by rocprofv3 in KiB; write bytes are estimated as GL2C_EA_WRREQ_64B_sum * 64. Time is summed across selected kernels and divided by 20 iterations.
