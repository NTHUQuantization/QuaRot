# Llama-3.1 8B Full-Model Evaluation

This directory contains the real-model evaluation harness for
`meta-llama/Llama-3.1-8B`.

## Token

This model is gated. Set `HF_TOKEN` after your HuggingFace account has access:

```bash
export HF_TOKEN=hf_xxx
```

## Current integration status

`fp16_hf` is a real HuggingFace full-model baseline.

`quarot_unfused` and `fused_quarot` now run through
`QuaRotLlamaForCausalLM`, a formal token-by-token wrapper around the HF
Llama-3.1 8B model. Prefill still uses the HF model and cache, then decode uses
the GQA-aware K1/K2/K3/FFN QuaRot paths.

The wrapper exposes:

- `prefill(input_ids, attention_mask=None, max_new_tokens=...)`
- `decode_one(input_ids, cache)`
- `generate(input_ids, attention_mask=None, max_new_tokens=...)`

This is not a monkey-patch of Transformers internals. Native
`transformers.generate()` / `LlamaDecoderLayer` cache replacement remains a
future integration step.

## Commands

```bash
python3 -m llama31_quarot.benchmark_full_model \
  --mode fused_quarot \
  --attn-implementation sdpa \
  --batches 1,2,4,8 \
  --context-lengths 10,128,1024,4096 \
  --iters 3 \
  --warmup 1 \
  --repeats 3 \
  --out-dir llama31_formal_latency_fused_quarot

python3 -m llama31_quarot.validate_quality \
  --reference-mode fp16_hf \
  --candidate-mode fused_quarot \
  --attn-implementation sdpa \
  --max-lengths 16,128,1024 \
  --max-new-tokens 8 \
  --out-dir llama31_formal_quality_fp16_vs_fused

python3 -m llama31_quarot.summarize_formal_full_model \
  --out llama31_formal_full_model_report_zh.md
```

The latest formal full-model report is `llama31_formal_full_model_report_zh.md`.
