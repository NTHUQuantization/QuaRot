#!/usr/bin/env bash
set -euo pipefail

mode=$1
out_dir=/workspace_root/fused_v1/qwen3_ti_td_comparison/memory_smoke_8b
prefix=${out_dir}/${mode//-/_}
result=${prefix}.json
csv=${prefix}_amd_smi.csv
log=${prefix}.log

mkdir -p "$out_dir"
cd /tmp
amd-smi monitor -p -t -u -m -v -w 1 --csv --file "$csv" &
monitor_pid=$!
cleanup() {
    kill -INT "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup EXIT

PYTHONPATH=/workspace_root/fused_v1:/workspace_root/fused_v1/third-party/hadacore \
TRITON_CACHE_DIR=/tmp/qwen3-8b-memory-smoke-triton \
QUAROT_FUSED_K1=0 \
python -m e2e.benchmark_pard2 \
    --mode "$mode" \
    --dataset humaneval \
    --limit 1 \
    --generated-tokens 32 \
    --max-cache-len 8192 \
    --warmups 1 \
    --sweeps 1 \
    --benchmark-profile qwen3_8b \
    --target /workspace_root/qwen3_8b_fused_v1_rtn_w4a4kv4 \
    --draft /workspace_root/.hf_cache/pard/hub/models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135 \
    --tokenizer /workspace_root/.hf_cache/pard/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
    --compile-mode max-autotune-no-cudagraphs \
    --output "$result" >"$log" 2>&1

cleanup
trap - EXIT
PYTHONPATH=/workspace_root/fused_v1 \
python /workspace_root/fused_v1/e2e/attach_amd_smi_monitor.py \
    --result "$result" --csv "$csv" --gate-percent 95
