#!/usr/bin/env bash
set -euo pipefail

result_dir=${1:-/workspace_root/fused_v1/qwen3_14b_results/formal_8k_ti_shared_drafter_20260909}
runner=/workspace_root/fused_v1/e2e/run_qwen3_14b_formal_8k.sh
idle_vram_bytes=$((1 << 30))

if [[ -e "$result_dir" ]]; then
    echo "result directory already exists: $result_dir" >&2
    exit 73
fi

while true; do
    used=$(rocm-smi --showmeminfo vram 2>/dev/null |
        awk '/VRAM Total Used Memory \(B\)/ {print $NF; exit}')
    if [[ -n "$used" && "$used" -lt "$idle_vram_bytes" ]]; then
        break
    fi
    echo "waiting_for_gpu used_bytes=${used:-unknown} at=$(date -Is)"
    sleep 30
done

for dataset in humaneval gsm8k math_500; do
    echo "starting corrected TI dataset=$dataset at=$(date -Is)"
    "$runner" pard2-ti "$dataset" "$result_dir"
    echo "completed corrected TI dataset=$dataset at=$(date -Is)"
done

echo "ALL_CORRECTED_TI_COMPLETE result_dir=$result_dir at=$(date -Is)"
