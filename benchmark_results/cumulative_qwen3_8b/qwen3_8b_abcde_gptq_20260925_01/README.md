# Qwen3-8B A–E benchmark: DE

Run ID: `qwen3_8b_abcde_gptq_20260925_01`. This branch owns variant(s) **DE**.

Read [the full report](formal/report_zh.md). A=HIP (source branch main), B=GEMM,
C=Hadacore (GEMM + HadaCore), D=Fusion, E=PARD2 (TI).
The ablation narrative is HIP -> GEMM -> Hadacore -> Fusion -> PARD2.
B adds only optimized GEMM, retaining HIP Hadamard; C retains B GEMM and
adds HadaCore. Both use the new 2026-09-26 commits, with fresh pilot runs.
Superseded pilot artifacts are archived separately with their original source identities.
Cross-variant output differences remain in all statistics. Ratios with differing
outputs describe natural-generation elapsed time, not equal-output pure speedups.

`formal/variants/` and `pilot/variants/` contain this branch's raw runs, manifests, compatibility patches,
GPU snapshots and timing/memory records. `formal/` and `pilot/` contain all five
variants' raw records and derived tables, making cross-variant comparisons auditable.
The formal workload has 96 prompts (32 per dataset), 3 repeats, natural EOS,
256 output-token cap, greedy, thinking disabled and batch size 1.
Pilot is separate: 12 disjoint prompts, cap 128.

P10/P90/P99 in primary tables use 32 per-prompt medians, linear interpolation.
Pooled-repeat percentiles are separately named. P99 is descriptive with this small
sample size. Peak allocated/reserved memory uses PyTorch counters reset per phase;
resident model memory is included. It is not whole-device VRAM usage.

Reproduction:

1. Use the recorded ROCm/PyTorch environment in `environment.json` and the exact
   source commit in `formal/variants/<variant>/variant_manifest.json`.
2. Overlay `runtime_source/<variant>/` onto that checkout. For D/E this snapshot
   preserves the actual measured local runtime independently of the result-only
   publication commit. No weights or compiled binaries are bundled.
3. Build the HIP extension using the branch's `setup.py build_ext --inplace`,
   `MAX_JOBS=2`. Build flags and binary SHA256 are recorded in build provenance
   and variant manifests. D/E also require the bundled HadaCore source build.
4. Supply the exact checkpoint, tokenizer and (for E) drafter matching the hashes
   in `protocol_manifest.json`. Do not requantize or substitute weights.
5. Set `OMP_NUM_THREADS=4`, `QUAROT_BATCHED_H128=1`,
   `QUAROT_STATIC_KV_METADATA=1`, `QUAROT_VERIFICATION_GRAPH=1`,
   `QUAROT_CHUNK_PREPROCESS=1`; run one GPU worker at a time on an idle GPU.
   Add D/E's `third-party/hadacore` directory to PYTHONPATH. Ensure Git trusts the
   explicitly selected checkout when running through a container mount.
6. Run `python harness/replay.py --variant A --repo /path/to/checkout
   --target /path/to/target --tokenizer /path/to/tokenizer
   --reference /path/to/this/folder
   --stage formal --output /path/to/new/replay_root` (one line).
   Substitute the desired variant; E additionally needs `--draft /path/to/draft`.

This run uses only GPTQ measurements. Old RTN results are separately archived;
no old C/D-labeled raw rows are mixed into this run. Current labels are A–E.
D/E use the same GPTQ target, and D supplies the paired AR oracle for E.
The PARD2 drafter remains BF16, as in the fixed test configuration.
For replay, first measure D, then reuse the same replay root for other variants.
The replay writes `results/<stage>/<variant>/` under that root and uses its
completed D records as the GPTQ oracle.

`checksums.json` covers every bundled file except itself. All validation precedes
publication. Publication adds only this result folder, without modifying kernels.
