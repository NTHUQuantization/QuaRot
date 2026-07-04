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
| fp16_hf | 1 | 10 | prefill_ms | 42.01 | 41.35 | 1.201 | 42.99 | 43.19 |
| fp16_hf | 1 | 10 | decode_ms_per_token | 40.04 | 39.97 | 0.2964 | 40.28 | 40.32 |
| fp16_hf | 1 | 128 | prefill_ms | 66.62 | 66.48 | 0.5676 | 67.09 | 67.17 |
| fp16_hf | 1 | 128 | decode_ms_per_token | 41.3 | 40.87 | 0.7679 | 41.92 | 42.06 |

## Quality / Logits Summary

| reference_mode | candidate_mode | prompt_id | max_length | max_error | mean_error | top1_match | top10_overlap | kl_ref_to_out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fp16_hf | fp16_hf | 0 | 16 | 0 | 0 | 1 | 1 | -1.885e-07 |
| fp16_hf | fp16_hf | 0 | 128 | 0 | 0 | 1 | 1 | -1.885e-07 |
| fp16_hf | fp16_hf | 1 | 16 | 0 | 0 | 1 | 1 | -1.544e-07 |
| fp16_hf | fp16_hf | 1 | 128 | 0 | 0 | 1 | 1 | -1.544e-07 |
| fp16_hf | fp16_hf | 2 | 16 | 0 | 0 | 1 | 1 | -2.083e-07 |
| fp16_hf | fp16_hf | 2 | 128 | 0 | 0 | 1 | 1 | -2.083e-07 |

## Limitations

- Llama-3.1 8B uses GQA (`q_heads=32`, `kv_heads=8`). Current K1/K2 prototypes assume identical Q/KV head count, so fused full-model attention must wait for GQA-aware kernels.
- `fp16_hf` is the first real-model baseline. QuaRot unfused/fused full-model modes intentionally fail fast until the GQA kernel gap is closed.
- This report should replace synthetic single-layer numbers only after full-model QuaRot modes are implemented and rerun.
