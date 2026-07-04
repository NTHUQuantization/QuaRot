# Single Decoder Layer Benchmark Summary

## Correctness

| batch | seq_len | ffn_hidden | variant | reference | max_error | mean_error | mean_relative_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 11008 | fp16_baseline | fp16_baseline | 0 | 0 | 0 |
| 1 | 10 | 11008 | quarot_unfused | fp16_baseline | 11.9766 | 2.48863 | 249.769 |
| 1 | 10 | 11008 | quarot_unfused | quarot_unfused | 0 | 0 | 0 |
| 1 | 10 | 11008 | fused_quarot | fp16_baseline | 11.6406 | 2.48861 | 399.48 |
| 1 | 10 | 11008 | fused_quarot | quarot_unfused | 0.946289 | 0.21026 | 95.8711 |

## Latency

| batch | seq_len | ffn_hidden | variant | latency_ms |
| --- | --- | --- | --- | --- |
| 1 | 10 | 11008 | fp16_baseline | 1.16264 |
| 1 | 10 | 11008 | quarot_unfused | 4.63988 |
| 1 | 10 | 11008 | fused_quarot | 1.04037 |
