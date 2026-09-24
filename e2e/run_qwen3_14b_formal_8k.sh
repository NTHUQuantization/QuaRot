#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <ar|pard2-ti|pard2-td> <humaneval|gsm8k|math_500> <output-dir>" >&2
    exit 2
fi

mode=$1
dataset=$2
output_dir=$3
case "$mode" in
    ar|pard2-ti|pard2-td) ;;
    *) echo "invalid mode: $mode" >&2; exit 2 ;;
esac
case "$dataset" in
    humaneval|gsm8k|math_500) ;;
    *) echo "invalid dataset: $dataset" >&2; exit 2 ;;
esac

target=/workspace_root/qwen3_14b_fused_v1_rtn_w4a4kv4
draft_ti=/workspace_root/.hf_cache/pard/hub/models--amd--PARD2-Qwen3-8B/snapshots/67a1516c8f6fc145cda99916799a0cbb3a4af135
draft_td=/workspace_root/.hf_cache/pard/models--amd--PARD2-Qwen3-14B/snapshots/679eff0b65ffaf5abd2dadd21a17909562935798
tokenizer=/workspace_root/.hf_cache/pard/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
hip_sha=5d5c26a9e0eb5efce924b70830821473ec9a142e3903f67ca104c58f299f921e
safe_mode=${mode//-/_}
out=${output_dir}/${safe_mode}_${dataset}
csv=${out}_amd_smi.csv
log=${out}.log
gate_log=${out}_safety_gate.log

mkdir -p "$output_dir"
if [[ -e "${out}.json" || -e "$csv" || -e "$log" ]]; then
    echo "refusing to overwrite an existing artifact for ${safe_mode}_${dataset}" >&2
    exit 73
fi

git config --global --add safe.directory /workspace_root/fused_v1
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
    --mode "$mode"
    --dataset "$dataset"
    --target "$target"
    --tokenizer "$tokenizer"
    --generated-tokens 256
    --max-cache-len 8192
    --warmups 8
    --sweeps 3
    --benchmark-profile qwen3_14b
    --expected-hip-sha256 "$hip_sha"
    --output "${out}.json"
)
if [[ "$mode" == ar ]]; then
    args+=(--compile-mode eager)
else
    if [[ "$mode" == pard2-ti ]]; then
        draft=$draft_ti
    else
        draft=$draft_td
    fi
    args+=(
        --draft "$draft"
        --compile-mode max-autotune-no-cudagraphs
        --precompile-draft-shapes
    )
    if [[ "$mode" == pard2-td ]]; then
        args+=(--td-basis-fold)
    fi
fi

TRITON_CACHE_DIR=/tmp/qwen3-14b-formal-triton-cache \
QUAROT_FUSED_K1=0 \
PYTHONUNBUFFERED=1 \
PYTHONPATH=/workspace_root/fused_v1:/workspace_root/fused_v1/third-party/hadacore \
python -m e2e.benchmark_pard2 "${args[@]}" >"$log" 2>&1 &
benchmark_pid=$!

gate_hit=0
while kill -0 "$benchmark_pid" 2>/dev/null; do
    if [[ -s "$csv" ]]; then
        line=$(tail -n 1 "$csv")
        IFS=, read -r timestamp gpu xcp power max_power hotspot memory rest <<<"$line"
        vram_percent=${line##*,}
        if awk -v v="$vram_percent" -v h="$hotspot" -v m="$memory" \
            'BEGIN { exit !((v+0)>=95.0 || (h+0)>=110.0 || (m+0)>=108.0) }'; then
            printf 'gate_hit timestamp=%s vram_percent=%s hotspot=%s memory=%s\n' \
                "$(date -Is)" "$vram_percent" "$hotspot" "$memory" >"$gate_log"
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
if [[ $status -ne 0 ]]; then
    exit "$status"
fi

PYTHONPATH=/workspace_root/fused_v1 \
python /workspace_root/fused_v1/e2e/attach_amd_smi_monitor.py \
    --result "${out}.json" --csv "$csv" --gate-percent 95
