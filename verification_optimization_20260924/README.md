# PARD-2 verification optimization reproduction

Run commands **serially**, with no other GPU job. The scripts retain the existing
benchmark preflight (refuse >1 GiB preexisting device allocation); inspect GPU
utilization and process lists as well before beginning. Never kill another job.

The existing container is `qwen3_32b_quarot_clean`; repository mount is
`/workspace_root/fused_v1`.
The corresponding host repo is
`/user/undergraduate/wfching25/HIP_Fusion/fused_v1`.
See the [execution/path guide](../e2e/FUSED_ONLY_AND_PARD2_ZH.md) for model,
tokenizer, dataset paths and fused-only generation with `--mode ar`.

Build and run the correctness gates before measuring:

```bash
set -e
cd /workspace_root/fused_v1
export PYTHONPATH="$PWD:$PWD/third-party/hadacore"
MAX_JOBS=4 python setup.py build_ext --inplace \
  > verification_optimization_20260924/final_build.log 2>&1
python -m pytest -q tests/test_verification_preprocess.py \
  tests/test_verification_norm_quant.py tests/test_rms_norm_quant_i4.py \
  tests/test_verification_metadata.py tests/test_verification_dispatch.py \
  tests/test_speculative.py tests/test_pard2_gpu.py tests/test_pard2_benchmark_contract.py \
  > verification_optimization_20260924/final_tests.log 2>&1
```

The tested configuration can be launched with
`bash verification_optimization_20260924/run_optimized_pard2.sh --mode pard2-ti`
(or `pard2-td`; use `ar` for fused-only generation without a drafter).
Each environment switch can be overridden independently;
passing `--no-fused-norm-quant` restores the norm fallback.

```bash
docker exec qwen3_32b_quarot_clean bash -lc '
  set -e
  cd /workspace_root/fused_v1
  export PYTHONPATH="$PWD:$PWD/third-party/hadacore"
  python verification_optimization_20260924/run_ab.py \
    --output verification_optimization_20260924 \
    --stages baseline had_only norm_only metadata_only graph_with_metadata \
      chunk_only all had_norm had_norm_metadata had_norm_metadata_graph baseline_repeat \
    --contexts 127 128 129 --samples 6 \
    --generated-tokens 32 --limit 1 --warmups 1 --sweeps 2 --compile-mode eager
  python verification_optimization_20260924/run_ab.py --stages baseline \
    --parts pard2 --modes ar --output verification_optimization_20260924 \
    --generated-tokens 32 --limit 1 --warmups 1 --sweeps 2 --compile-mode eager
  python verification_optimization_20260924/summarize.py --baseline baseline_repeat
  python verification_optimization_20260924/write_report.py
'
```

`run_ab.py --parts target` and `--parts pard2` allow scheduling primitive tests
between measurement phases. `--modes ar` records the same prompt's greedy target
oracle. `--skip-complete` skips existing result files; use only for previously
successful stages with unchanged settings. Use a fresh output directory when
changing the protocol; incomplete or failed results are rerun. All flags are explicit,
and each stage writes `manifest.json`
and complete subprocess logs. A nonzero subprocess return aborts the sweep.
Residual-add fusion is measured separately by the primitive benchmark only;
it is not integrated into the model path and has no PARD-2 throughput claim.

|Stage|Batched H128|Fused norm/quant|Static metadata|Verification graph|Q/KV chunk preprocessing|
|---|---:|---:|---:|---:|---:|
|baseline|0|0|0|0|0|
|had_only|1|0|0|0|0|
|norm_only|0|1|0|0|0|
|metadata_only|0|0|1|0|0|
|graph_with_metadata|0|0|1|1|0|
|chunk_only|0|0|0|0|1|
|had_norm|1|1|0|0|0|
|had_norm_metadata|1|1|1|0|0|
|had_norm_metadata_graph|1|1|1|1|0|
|all|1|1|1|1|1|
|baseline_repeat|0|0|0|0|0|

Graph's isolated effect is `graph_with_metadata / metadata_only`, because fixed
metadata is a required dependency. The cumulative graph comparison is
`had_norm_metadata_graph / had_norm_metadata`. Chunk preprocessing may replace
the Q Hadamard path itself, so its independent and cumulative effects must both
be measured; launch reductions cannot be added arithmetically.

`target_probe.py` loads the same exact-row RMSNorm wrappers used by PARD-2, with
no drafter or TD feature collector. For M=1/15/16 at contexts127/128/129 it records
GPU trace events, kernel histograms, event/wall samples, graph capture
costs, finite logits, exact packed KV/scale parity with sequential input,
argmax parity, strict future-token causal checks, reject-all and partial-reject
overwrite checks. Optimization cases additionally compare packed KV/scales and
argmax against `baseline/target/artifacts.pt`. Graph output tensors are cloned
before replay to avoid accidentally comparing aliases. Both M15 and M16 are
warmed separately before profiling and latency sampling. ROCTracer on this stack
omits a variable subset of graph child kernels: graph counts therefore use
read-only `hipGraphGetNodes` / `hipGraphNodeGetType` enumeration, recursively
including child graphs, plus external direct launch APIs. Partial trace-event
counts remain in separate fields. A graph's internal GPU kernels still count
individually; a replay API call is not one GPU kernel.

`e2e/benchmark_pard2.py` performs real TI/TD generation with unchanged checkpoint,
dataset, tokenizer, quantization and greedy verification rules. Its output keeps
the existing prompt/token hashes, output token IDs, acceptance sequence, per-stage
timings and memory/provenance contract. Additional provenance records optimization
flags and graph captures before and after measured sweeps. `summarize.py` compares
paired output IDs and acceptance lengths, and computes verify ms per real step.

The default one-prompt/32-token results are **smoke diagnostics**, not the formal
80/80/20-prompt three-dataset qualification. The official benchmark retains its
formal commands and gates; these scripts never change thresholds. Raw-audit
results use the historical plain-model configuration and are kept separately from
the exact-runtime baseline because their launch counts and norm semantics differ.

`artifacts.pt` and trace files are reproducibility artifacts; unused cache storage
is excluded from comparisons. An absent measurement is reported as absent, never
as zero latency or estimated tokens/s.

## Tracked evidence and local artifacts

The report's per-stage `target/target.json`, TI/TD result JSON, baseline AR oracle,
manifests, idle preflight records, `summary.json`, `norm_primitive.json`, and final
build/test logs are tracked. The small rsqrt diagnostic source/IR and two annotated
pre-hardening source snapshots preserve the numerical investigation and the
initial A/B implementation. They are historical fixtures, not imported runtime code.
Large `*.trace.json`, `artifacts.pt`, compiler binaries, and intermediate logs stay
local; all can be recreated with the commands above on the documented environment.

To regenerate the committed tables without launching GPU work, from repo root:

```bash
python verification_optimization_20260924/summarize.py --baseline baseline_repeat
python verification_optimization_20260924/write_report.py
```

To rerun measurements, use a fresh output directory rather than overwrite the
historical evidence. `summarize.py --root NEW_DIRECTORY --baseline baseline_repeat`
can summarize that run; `write_report.py` is specific to the original completed
matrix beside the script. The original `m16_tile_audit_20260924/` report and traces
are local audit inputs, excluded by `.gitignore`; the measured non-GEMM findings
and final A/B counts are preserved in this directory's committed report and JSON.

The separate [AR versus PARD-2 comparison](../pard2_speedup_20260924/README.md)
has not completed: its first attempt timed out waiting for an idle GPU. The
speedups in this directory compare PARD-2 configurations, not PARD-2 versus AR.
