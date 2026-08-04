#!/usr/bin/env bash
set -euo pipefail

QUAROT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HIP_ROOT=$(cd "$QUAROT_ROOT/.." && pwd)
IMAGE=${PARD_DOCKER_IMAGE:-rocm/vllm-dev:rocm7.2_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.14.0rc0}
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)
ENV_ARGS=()
if [[ -f "$QUAROT_ROOT/.hf_env" ]]; then
  ENV_ARGS+=(--env-file "$QUAROT_ROOT/.hf_env")
fi

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env USER=pard \
  --env LOGNAME=pard \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add "$VIDEO_GID" \
  --group-add "$RENDER_GID" \
  --ipc=host \
  -v "$HIP_ROOT:/workspace/HIP_Fusion" \
  -w /workspace/HIP_Fusion/QuaRot \
  "${ENV_ARGS[@]}" \
  -e HOME=/tmp/pard-home \
  -e XDG_CACHE_HOME=/workspace/HIP_Fusion/QuaRot/.pard_compile_cache \
  -e TRITON_CACHE_DIR=/workspace/HIP_Fusion/QuaRot/.pard_compile_cache/triton \
  -e HF_HOME=/workspace/HIP_Fusion/.hf_cache/pard \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/HIP_Fusion/QuaRot/.pard_compile_cache \
  "$IMAGE" \
  bash -c './pard_benchmark/setup_env.sh && exec ./.venv-pard/bin/python -m pard_benchmark.matrix "$@"' bash "$@"
