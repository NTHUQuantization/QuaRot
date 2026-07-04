# Llama-3.1 8B Full Model / QuaRot Fusion 評估報告

## Model / Integration Status

- model_id: `meta-llama/Llama-3.1-8B`
- hidden_size: `4096`
- intermediate_size: `14336`
- layers: `32`
- q_heads: `32`
- kv_heads: `8`
- head_dim: `128`
- current fused full-model compatibility: `False`
- note: Llama-3.1 8B uses GQA (q_heads=32, kv_heads=8). Current K1/K2 HIP prototypes assume q_heads == kv_heads in paged KV decode.

## Latency Summary

| mode | batch | context_len | metric | mean | median | std | p90 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16_hf | 1 | 10 | prefill_ms | 41.57 | 41.13 | 0.8195 | 42.46 | 42.7 |
| fp16_hf | 1 | 10 | decode_ms_per_token | 41.93 | 40.61 | 3.038 | 44.73 | 46.04 |
| fp16_hf | 1 | 128 | prefill_ms | 65.57 | 65.54 | 0.1322 | 65.72 | 65.74 |
| fp16_hf | 1 | 128 | decode_ms_per_token | 40.99 | 40.76 | 0.506 | 41.56 | 41.6 |
| fp16_hf | 1 | 1024 | prefill_ms | 205.1 | 205.2 | 0.4446 | 205.6 | 205.6 |
| fp16_hf | 1 | 1024 | decode_ms_per_token | 43.36 | 42.84 | 0.9292 | 44.39 | 44.43 |

## Quality / Logits Summary

No quality summary found.

## Limitations

- Llama-3.1 8B uses GQA (`q_heads=32`, `kv_heads=8`). Current K1/K2 prototypes assume identical Q/KV head count, so fused full-model attention must wait for GQA-aware kernels.
- `fp16_hf` is the first real-model baseline. QuaRot unfused/fused full-model modes intentionally fail fast until the GQA kernel gap is closed.
- This report should replace synthetic single-layer numbers only after full-model QuaRot modes are implemented and rerun.
