# Single Decoder Layer Ablation

| batch | seq_len | ffn_hidden | variant | latency_ms | speedup_vs_unfused |
| --- | --- | --- | --- | --- | --- |
| 1 | 10 | 11008 | quarot_unfused | 4.55569 | 1 |
| 1 | 10 | 11008 | k1_fused | 2.64454 | 1.72267 |
| 1 | 10 | 11008 | k1_k2_fused | 2.45013 | 1.85936 |
| 1 | 10 | 11008 | attention_fused | 1.71501 | 2.65636 |
| 1 | 10 | 11008 | fused_quarot | 1.02724 | 4.43486 |
