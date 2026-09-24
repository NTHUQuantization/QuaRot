# Fused PARD2 source index

Start with [PARD2_QWEN3.md](PARD2_QWEN3.md) for the current support matrix and commands.
For fused-only AR, current optimization switches, and exact host/container paths,
see [FUSED_ONLY_AND_PARD2_ZH.md](FUSED_ONLY_AND_PARD2_ZH.md).

| Area | Entry points |
|---|---|
| Generation and profile contracts | `pard2.py`, `speculative.py` |
| Benchmark and qualification | `benchmark_pard2.py`, `qualify_pard2.py` |
| Checkpoint conversion | `checkpoint_utils/quantize_checkpoint.py`, `streaming_rtn.py`, `streaming_gptq.py` |
| HIP shape dispatch | `../quarot/kernels/gemm.hip`, `bindings.cpp` |
| Model and kernel regression tests | `../tests/test_model_compat.py`, `test_fused_hip.py`, `test_pard2_gpu.py` |
| Profile/provenance regression tests | `../tests/test_speculative.py`, `test_qwen3_14b_profile.py`, `test_pard2_benchmark_contract.py`, `test_streaming_gptq.py` |
| Checkpoint audit and migration | `audit_qwen3_32b_checkpoint.py`, `repair_qwen3_32b_gptq_norms.py` |
| Distribution and task evaluation | `compare_qwen3_32b_output_distribution.py`, `score_qwen3_32b_tasks.py`, `run_humaneval_sandbox.py` |
| Monitoring and plots | `attach_amd_smi_monitor.py`, `summarize_qwen3_32b_td_monitor.py`, `plot_qwen3_*` |
| Kernel microbenchmark | `../benchmarks/qwen3_32b_kernel_benchmark.py` |

## Experimental 32B TD

`prepare_qwen3_32b_td_*`, `capture_qwen3_32b_td_contract.py`,
`qwen3_32b_td_features*`, `qwen3_32b_td_numeric.py`,
`qwen3_32b_td_teacher_cache.py`, `qwen3_32b_td_projection_train*`,
`qwen3_32b_td_top4_train_windowed.py`, `diagnose_qwen3_32b_td_feature_nan.py`,
and `benchmark_qwen3_32b_td_heldout*` preserve the data preparation,
feature alignment, training, and held-out evaluation workflow.
These tools do not establish canonical 32B TD support.

- [Experiment plan](QWEN3_32B_PARD2_TD_EXPERIMENT_PLAN_ZH.md) and
  [machine-readable matrix](qwen3_32b_td_experiment_matrix.json): original plan and frozen baseline paths.
- [S0–S2 results](QWEN3_32B_PARD2_TD_S0_S2_RESULTS_ZH.md).
- [S3–S5 status](QWEN3_32B_PARD2_TD_S3_S5_STATUS_ZH.md) and
  [results](QWEN3_32B_PARD2_TD_S3_S5_RESULTS_ZH.md).
- [32B W4A4KV4 report](QWEN3_32B_W4A4KV4_PARD2_REPORT_ZH.md).
- [HIP design comparison](QWEN3_HIP_DESIGN_COMPARISON_ZH.md).

Reports, experiment configuration, and `run_qwen3_*` shell scripts retain
historical absolute paths, model/binary pins, and artifact names. Check these
before reuse; use the current setup guide for portable CLI examples.

## Repository contents

Source code, regression tests, experiment plans, and written reports are tracked.
New checkpoint weights, training tensors, compiler caches, profiler traces,
monitor CSVs, and raw experiment directories stay local and are ignored.
The curated `verification_optimization_20260924` evidence is an explicit exception:
small per-stage result JSON, manifests, final build/test logs, and summary are
tracked so its tables can be regenerated without GPU execution. Large traces and
tensor artifacts remain local. Its correctness result is 229 targeted tests;
the 317-test result below belongs to the earlier release, not this change.
Older result artifacts already tracked by the branch remain available.
`third-party/PARD` is an optional local upstream checkout, not a Git submodule;
its pinned revision and benchmark data setup are documented in the setup guide.

## Release validation (2026-09-24)

The HIP extension was rebuilt from this source for `gfx1201` in the existing
ROCm container (PyTorch `2.9.1+git5bc97ba`, Transformers `4.57.6`). The full
`tests` suite passed: **317 passed**, with 17 warnings. Testing loaded the newly
built extension from a temporary build directory; the pre-existing local binary
was left in place. Python AST parsing, shell `bash -n`, and `git diff --check`
also passed. Full model quality/latency benchmark matrices were not rerun for
this source/documentation release; historical results retain their original scope.
