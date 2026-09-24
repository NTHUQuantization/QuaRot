#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <ar|pard2-ti> <humaneval|gsm8k|math_500> <output-dir>" >&2
    exit 2
fi

mode=$1
dataset=$2
output_dir=$3
case "$mode" in ar|pard2-ti) ;; *) echo "invalid mode: $mode" >&2; exit 2 ;; esac
case "$dataset" in humaneval|gsm8k|math_500) ;; *) echo "invalid dataset: $dataset" >&2; exit 2 ;; esac

target=/workspace_root/qwen3_32b_fused_v1_gptq_w4a4kv4_v1
draft=/workspace_root/.hf_cache/pard/hub/models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135
tokenizer=/workspace_root/.hf_cache/pard/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
hip_sha=8e544408498612b2eae27b6fce9a52e939bded735eba77772692f6aa5fe5cfc5
safe_mode=${mode//-/_}
out=${output_dir}/${safe_mode}_${dataset}
csv=${out}_amd_smi.csv
log=${out}.log
gate_log=${out}_safety_gate.log

mkdir -p "$output_dir"
cd /tmp
amd-smi monitor -p -t -u -m -v -w 1 --csv --file "$csv" &
monitor_pid=$!
benchmark_pid=

cleanup() {
    if [[ -n "${benchmark_pid:-}" ]]; then
        kill -INT "$benchmark_pid" 2>/dev/null || true
    fi
    if [[ -n "${monitor_pid:-}" ]]; then
        kill -INT "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

args=(
    --mode "$mode" --dataset "$dataset"
    --target "$target" --tokenizer "$tokenizer"
    --generated-tokens 256 --max-cache-len 8192
    --warmups 8 --sweeps 3 --compile-mode eager
    --benchmark-profile qwen3_32b
    --expected-hip-sha256 "$hip_sha"
    --output "${out}.json"
)
if [[ "$mode" != ar ]]; then
    args+=(--draft "$draft")
fi

TRITON_CACHE_DIR=/tmp/qwen3-32b-uid3016-triton-cache \
QUAROT_FUSED_K1=0 \
QUAROT_QWEN3_32B_GROUPED_NWAVES=4 \
QUAROT_QWEN3_32B_MULTI_NWAVES=2 \
PYTHONPATH=/tmp/qwen3-32b-ti-runtime-option1:/workspace_root/fused_v1:/workspace_root/fused_v1/third-party/hadacore \
python -m e2e.benchmark_pard2 "${args[@]}" > "$log" 2>&1 &
benchmark_pid=$!

gate_hit=0
while kill -0 "$benchmark_pid" 2>/dev/null; do
    if [[ -s "$csv" ]]; then
        line=$(tail -n 1 "$csv")
        IFS=, read -r timestamp gpu xcp power max_power hotspot memory rest <<< "$line"
        vram_percent=${line##*,}
        if awk -v v="$vram_percent" -v h="$hotspot" -v m="$memory" \
            'BEGIN { exit !((v+0)>=95.0 || (h+0)>=110.0 || (m+0)>=108.0) }'; then
            printf 'gate_hit timestamp=%s vram_percent=%s hotspot=%s memory=%s\n' \
                "$(date -Is)" "$vram_percent" "$hotspot" "$memory" > "$gate_log"
            kill -INT "$benchmark_pid" 2>/dev/null || true
            gate_hit=1
            break
        fi
    fi
    sleep 1
done

wait "$benchmark_pid"
status=$?
benchmark_pid=
kill -INT "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=
if [[ $gate_hit -eq 1 ]]; then
    exit 95
fi
if [[ $status -eq 0 ]]; then
    PYTHONPATH=/workspace_root/fused_v1 \
        python /workspace_root/fused_v1/e2e/attach_amd_smi_monitor.py \
        --result "${out}.json" --csv "$csv" --gate-percent 95
    attach_status=$?
    if [[ $attach_status -ne 0 ]]; then
        exit "$attach_status"
    fi
fi
exit "$status"
