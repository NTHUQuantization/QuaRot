#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <artifact-json> <log-prefix> <command...>" >&2
    exit 2
fi

artifact=$1
prefix=$2
shift 2
csv=${prefix}_amd_smi.csv
log=${prefix}.log
gate_log=${prefix}_safety_gate.log
mkdir -p "$(dirname "$prefix")"

amd-smi monitor -p -t -u -m -v -w 1 --csv --file "$csv" &
monitor_pid=$!
workload_pid=

cleanup() {
    if [[ -n "${workload_pid:-}" ]]; then
        kill -INT "$workload_pid" 2>/dev/null || true
    fi
    if [[ -n "${monitor_pid:-}" ]]; then
        kill -INT "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

"$@" >"$log" 2>&1 &
workload_pid=$!
gate_hit=0
while kill -0 "$workload_pid" 2>/dev/null; do
    if [[ -s "$csv" ]]; then
        line=$(tail -n 1 "$csv")
        IFS=, read -r timestamp gpu xcp power max_power hotspot memory rest <<< "$line"
        vram_percent=${line##*,}
        if awk -v v="$vram_percent" -v h="$hotspot" -v m="$memory" \
            'BEGIN { exit !((v+0)>=95.0 || (h+0)>=110.0 || (m+0)>=108.0) }'; then
            printf 'gate_hit timestamp=%s vram_percent=%s hotspot=%s memory=%s\n' \
                "$(date -Is)" "$vram_percent" "$hotspot" "$memory" >"$gate_log"
            kill -INT "$workload_pid" 2>/dev/null || true
            gate_hit=1
            break
        fi
    fi
    sleep 1
done

wait "$workload_pid"
status=$?
workload_pid=
kill -INT "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=
if [[ $gate_hit -eq 1 ]]; then
    exit 95
fi
if [[ $status -ne 0 ]]; then
    exit "$status"
fi
PYTHONPATH=/workspace_root/fused_v1 \
python /workspace_root/fused_v1/e2e/summarize_qwen3_32b_td_monitor.py \
    --result "$artifact" --csv "$csv" --gate-percent 95
