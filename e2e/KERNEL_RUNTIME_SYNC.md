# Kernel and runtime synchronization

This update makes the current fused kernel and matching runtime available directly
on `fused_pard2`. Rebuild the extension after updating the checkout:

```bash
QUAROT_HIP_ARCHS=gfx1201 python setup.py build_ext --inplace --force
```

Run this in the project's ROCm development environment. An existing `_HIP.so`
from an older checkout does not pick up source changes automatically.

The kernel uses wave-register H32/H64/H128/H256 transforms and grouped-256 FP32
butterflies with matching single-row and verification-row quantization. Fused
RoPE retains explicit FP16 rounding boundaries instead of contracting half-FMA.
The existing GEMM, norm, verification metadata and graph paths remain integrated.

The runtime includes phase measurement adapters, persistent cache handling for
batch shapes, and last-logit-only draft prefill. Natural EOS generation remains
batch 1. The separate synchronous PARD2 batch path requires fixed output length
(`ignore_eos=True`) and fixed draft_k, and commits the batch's shared accepted
prefix. It is not a dynamic per-request scheduler. Batched AR uses its dedicated
benchmark adapter.

`ti_zero_accept_fallback` is an explicit opt-in TI-to-AR policy, defaults to zero,
and is restricted to batch 1. Provenance distinguishes the hybrid from pure TI.
The 8B CLI's default target is now GPTQ and validates the default checkpoint's
shard signatures/index before use; pass model paths explicitly on another machine.

For independent named-version benchmarking with frozen inputs and comparison
checks, use the [portable benchmark guide](https://github.com/NTHUQuantization/QuaRot/blob/benchmark/qwen3-8b-ablation-gptq-handoff/benchmarks/qwen3_8b_benchmark/README.md).
Point its `repo` setting at this checkout and rebuild here.

## Validation

The synchronized runtime passed 91 CPU tests and 85 subtests, covering speculative
contracts, phase boundaries, synchronous batches and request-local fallback:

```bash
python -m pytest -q \
  tests/test_speculative.py tests/test_pard2_benchmark_contract.py \
  tests/test_benchmark_pard2_phases.py tests/test_synchronous_pard.py \
  tests/test_ti_fallback.py
```

The kernel source matches the previously qualified local source exactly. The
prior kernel integration recorded 365 GPU regression passes. GPU tests and a
fresh build were not repeated for this synchronization while a separate formal
measurement occupied the GPU. This update includes the strict numerical,
rowwise, stream and graph regression tests for rerunning on an idle GPU:

```bash
python -m pytest -q tests/test_fused_hip.py tests/test_pard_fix_kernels.py
```
