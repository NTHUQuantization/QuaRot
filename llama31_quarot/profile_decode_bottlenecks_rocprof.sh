#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-decode_bottleneck_profiling_results}"
ROCPROF_DIR="${OUT_DIR}/rocprof"
ROCPROF_ITERS="${ROCPROF_ITERS:-10}"
ROCPROF_WARMUP="${ROCPROF_WARMUP:-3}"
RUN_COUNTERS="${RUN_COUNTERS:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

mkdir -p "${ROCPROF_DIR}"

local_flag=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_flag+=(--local-files-only)
fi

variants=(unfused_INT4 fused_current fused_hadacore256)
shapes=("1 128" "1 4096" "4 1024")

for shape in "${shapes[@]}"; do
  read -r batch context <<<"${shape}"
  for variant in "${variants[@]}"; do
    workload="${variant}_B${batch}_L${context}"
    rocprofv3 \
      --selected-regions \
      --kernel-trace \
      --marker-trace \
      --memory-copy-trace \
      --stats \
      -f csv \
      -d "${ROCPROF_DIR}" \
      -o "${workload}" \
      -- python3 -m llama31_quarot.profile_decode_bottlenecks \
        --task rocprof_workload \
        --rocprof-variant "${variant}" \
        --rocprof-batch "${batch}" \
        --rocprof-context "${context}" \
        --rocprof-iters "${ROCPROF_ITERS}" \
        --rocprof-warmup "${ROCPROF_WARMUP}" \
        --out-dir "${OUT_DIR}" \
        "${local_flag[@]}"

    if [[ "${RUN_COUNTERS}" == "1" ]]; then
      rocprofv3 \
        --selected-regions \
        --pmc SQ_WAVES_sum GRBM_GUI_ACTIVE GRBM_COUNT CU_NUM SIMD_NUM \
        --stats \
        -f csv \
        -d "${ROCPROF_DIR}" \
        -o "${workload}_counters" \
        -- python3 -m llama31_quarot.profile_decode_bottlenecks \
          --task rocprof_workload \
          --rocprof-variant "${variant}" \
          --rocprof-batch "${batch}" \
          --rocprof-context "${context}" \
          --rocprof-iters "${ROCPROF_ITERS}" \
          --rocprof-warmup "${ROCPROF_WARMUP}" \
          --out-dir "${OUT_DIR}" \
          "${local_flag[@]}"
    fi
  done
done

if [[ "${RUN_COUNTERS}" == "1" ]]; then
  probe_sets=(
    "FetchSize"
    "GL2C_EA_RDREQ_64B_sum GL2C_EA_WRREQ_64B_sum"
    "SQ_INSTS_LDS SQC_LDS_IDX_ACTIVE SQC_LDS_BANK_CONFLICT"
    "MeanOccupancyPerCU"
  )
  probe_index=0
  for counters in "${probe_sets[@]}"; do
    read -ra counter_args <<<"${counters}"
    rocprofv3 \
      --selected-regions \
      --pmc "${counter_args[@]}" \
      --stats \
      -f csv \
      -d "${ROCPROF_DIR}" \
      -o "fused_current_B1_L128_probe${probe_index}" \
      -- python3 -m llama31_quarot.profile_decode_bottlenecks \
        --task rocprof_workload \
        --rocprof-variant fused_current \
        --rocprof-batch 1 \
        --rocprof-context 128 \
        --rocprof-iters "${ROCPROF_ITERS}" \
        --rocprof-warmup "${ROCPROF_WARMUP}" \
        --out-dir "${OUT_DIR}" \
        "${local_flag[@]}"
    probe_index=$((probe_index + 1))
  done
fi
