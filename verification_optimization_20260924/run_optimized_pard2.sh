#!/usr/bin/env bash
# Run inside the existing ROCm environment. Later CLI options override defaults.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/third-party/hadacore${PYTHONPATH:+:$PYTHONPATH}"
export QUAROT_BATCHED_H128="${QUAROT_BATCHED_H128:-1}"
export QUAROT_STATIC_KV_METADATA="${QUAROT_STATIC_KV_METADATA:-1}"
export QUAROT_VERIFICATION_GRAPH="${QUAROT_VERIFICATION_GRAPH:-1}"
export QUAROT_CHUNK_PREPROCESS="${QUAROT_CHUNK_PREPROCESS:-1}"
exec python -m e2e.pard2 --compile-mode eager --fused-norm-quant "$@"
