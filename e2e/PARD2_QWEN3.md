
# fused_v1 × PARD2-Qwen3-8B

Use `e2e/pard2.py` for generation and `e2e/benchmark_pard2.py` for
isolated per-mode benchmarks. The benchmark verifies the pinned dataset hashes,
refuses a GPU with more than 1 GiB pre-existing use, and marks any
`--limit`/`--offset` run as `qualified: false`.

The default phase-1 path uses native GQA KV4, the unified prefill/decode/chunk
writer, transactional logical cache lengths, selected TD taps, and the official
BF16 drafter. A 64-token one-prompt HumanEval smoke (not a formal result)
produced exact AR/TD token parity, 1.75x steady speedup, 1.48x E2E speedup,
mean accepted length 4.27, and 8.19 GiB peak allocated VRAM.

Phase-2 calibration must be regenerated from this runtime's tune features:

```bash
python e2e/pard2_collect_features.py \
  --prompts-per-dataset 4 \
  --output /path/to/qwen3_tune_features.pt
python e2e/pard2_calibrate.py \
  --tune-features /path/to/qwen3_tune_features.pt \
  --output /path/to/qwen3_calibration.pt
```

No phase-2 option is enabled by default. The experimental M16 BPre geometry is
available with `QUAROT_ENABLE_M16_BPRE=1`, but its one-prompt E2E gain was
below the required 3%. The generated W4A4 drafter at
`/user/undergraduate/wfching25/HIP_Fusion/pard2_qwen3_8b_drafter_fused_v1_rtn_w4a4`
failed the quality gate (mean accept length 1.0 and zero accepted draft tokens)
and is retained only as an ablation artifact. A three-prompt calibration smoke
improved steady latency on one held-out prompt but did not improve E2E
throughput, so it is also not adopted.

The three-mode MATH-500 formal run has completed, but TI/TD do not yet have
exact greedy parity with AR, so the full 80/80/20 matrix remains fail-fast
incomplete and no adoption or literature-attainment claim is valid. See
[`PARD2_INTEGRATION_REPORT_ZH.md`](PARD2_INTEGRATION_REPORT_ZH.md) for the
architecture, implementation, experiments, formal results, and next steps.

## Resources and model

The first integration target is now dense `Qwen/Qwen3-8B` in no-thinking,
batch-one greedy mode.  This model is registered by fused_v1's Qwen3 runtime
and has an official matching PARD2 drafter.

Pinned resources:

- Target source: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`
- PARD2 drafter: `amd/PARD2-Qwen3-8B@67a1516c8f6fc145cda99916799a0cbb3a4af135`
- AMD PARD source: `6f279bf3f1680e0b5d71c562ca5b91bdeef4c038`

The PARD2 drafter, projection, target source and tokenizer are present in the
shared Hugging Face cache and pass the Hub checksum manifest.  The converted
W4A4KV4 target is available at
`/user/undergraduate/wfching25/HIP_Fusion/qwen3_8b_fused_v1_rtn_w4a4kv4`.

To regenerate a self-contained target (including PARD2 rotation/final-norm
metadata), use:

```bash
python e2e/checkpoint_utils/quantize_checkpoint.py \
  --model Qwen/Qwen3-8B \
  --output /path/to/qwen3_8b_fused_v1_rtn_w4a4kv4 \
  --w-rtn \
  --rotation-device cuda \
  --rotation-dtype float32
```

Pass that directory through `--target`.  AR, PARD2-TI and PARD2-TD then share
the same W4A4KV4 checkpoint and cache/kernel implementation.

The official AMD vLLM-v1 Qwen3-8B TD reference reports 6.75x on HumanEval and
6.44x on GSM8K.  It does not report Qwen3 MATH-500 or acceptance length in that
table, so the qualification tool deliberately does not fabricate those
literature fractions.
