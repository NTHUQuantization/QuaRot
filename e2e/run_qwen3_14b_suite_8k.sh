#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <new-result-dir>" >&2
    exit 2
fi

result_dir=$1
runner=/workspace_root/fused_v1/e2e/run_qwen3_14b_formal_8k.sh
if [[ -e "$result_dir" ]]; then
    echo "refusing to reuse existing result directory: $result_dir" >&2
    exit 73
fi

for dataset in humaneval gsm8k math_500; do
    for mode in ar pard2-ti pard2-td; do
        echo "starting mode=$mode dataset=$dataset"
        "$runner" "$mode" "$dataset" "$result_dir"
        echo "completed mode=$mode dataset=$dataset"
    done
done

PYTHONPATH=/workspace_root/fused_v1:/workspace_root/fused_v1/third-party/hadacore \
/opt/venv/bin/python -m e2e.qualify_pard2 \
    --result-dir "$result_dir" \
    --phase 1 \
    --benchmark-profile qwen3_14b \
    --vram-gate-percent 90 \
    --output "$result_dir/qualification.json"

/opt/venv/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("hard_gate") is True and p.get("all_exact_parity") is True else 1)' \
    "$result_dir/qualification.json"
