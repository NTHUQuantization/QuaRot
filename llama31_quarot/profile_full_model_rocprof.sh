#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-fp16_hf}"
OUT_DIR="${2:-llama31_full_model_rocprof}"
BATCH="${BATCH:-1}"
CONTEXT_LEN="${CONTEXT_LEN:-128}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"

mkdir -p "${OUT_DIR}"

rocprofv3 --kernel-trace --stats -f csv \
  -d "${OUT_DIR}" \
  -o "${MODE}_B${BATCH}_L${CONTEXT_LEN}" \
  -- python3 -m llama31_quarot.benchmark_full_model \
    --mode "${MODE}" \
    --batches "${BATCH}" \
    --context-lengths "${CONTEXT_LEN}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}" \
    --repeats 1 \
    --out-dir llama31_full_model_results

