# RDNA4 QuaRot W4A4KV4 runtime

This directory is an isolated packed-weight runtime for
`meta-llama/Llama-3.1-8B-Instruct`. It does not use the legacy HF FP16 linear
modules or the K3/FFN dequantization path.

## Implemented contract

- Signed symmetric group-128 W4/A4 packing with two's-complement nibbles.
- Linear layout `[N/16, K/128, 16, 64]` and FP16 `[N, K/128]` scales.
- Seed-0 residual rotation, RMSNorm folding, head-wise H128 V folding, H4096 O
  folding, and H14336 down-projection folding.
- Packed W4 embedding, merged QKV and gate/up, O/down, and LM-head modules.
- Native HIP activation pack, packed INT8-dot W4A4 linear, and embedding lookup.
- Decode fusions for RMSNorm+A4, SiLU×up+H14336+A4, QKV split+RoPE+H128,
  cross-head H32, and direct biased-nibble KV4 page append.
- FP16 causal prefill, paged FlashInfer-compatible KV4 decode, and B×K virtual
  request speculative verification with commit/rollback semantics.
- Sharded safetensors schema and a release audit that rejects FP16 shadow
  tensors and checkpoints larger than 4.5 GiB.

RMSNorm, RoPE, attention, SiLU, residuals, accumulation, and logits remain
FP16/FP32 by design.

## Build and test

```bash
cd w4a4_kernels
python setup.py build_ext --inplace
cd ..
PYTHONPATH=. python -m pytest -q tests/test_w4a4_runtime.py
```

The HIP test is skipped when no ROCm GPU is visible. Build success alone is not
a gfx1201 numerical or performance qualification.

## Bring-up checkpoint

```bash
PYTHONPATH=. python -m w4a4_runtime.export_checkpoint \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-revision REVISION \
  --local-files-only \
  --quant-method rtn \
  --output /path/to/llama31-8b-quarot-rtn-w4a4kv4
```

RTN is intentionally marked as RTN. The exporter fails if `--quant-method
gptq` is requested until the WikiText-2 Hessian calibration path is complete;
it never labels RTN weights as GPTQ.

Load and execute the packed model with:

```python
from w4a4_runtime import QuaRotW4A4LlamaForCausalLM, audit_checkpoint

print(audit_checkpoint(checkpoint_path))
model = QuaRotW4A4LlamaForCausalLM.from_quantized(checkpoint_path, device="cuda")
logits, cache = model.prefill(input_ids, max_new_tokens=128)
logits, cache = model.decode_one(next_token, cache)
logits, transaction = model.verify_chunk(draft_tokens, cache)
transaction.commit(accepted_length)
```

The loader constructs modules on the meta device and installs packed buffers
directly, avoiding a second resident copy of the approximately 4 GiB weights.

## Qualification status

The gfx1201 M=1 path consumes W4 and A4 directly, unpacks each four-value tile
into signed INT8 registers, uses RDNA packed INT8 dot instructions with INT32
partial sums, applies group scales in FP32, and writes FP16. R9700 autotuning
selects two outputs per wave. Decode KV4 is quantized and written directly to
the destination page/slot without temporary tensors or PyTorch index kernels.

Speculative verification uses a measured hybrid dispatch. M=2–8 uses the
shared-weight packed-dot row kernel; M=9–16 uses a padded rocWMMA signed-I8
16x16x16 kernel with INT32 accumulation. The M4/M8/M16 rocWMMA variants remain
selectable through `W4A4_PARALLEL_KERNEL=wmma` for ablation, while `auto` is the
production default. Provisional K/V quantization and page append are fused into
one launch per layer, and virtual-request metadata is built in one device
launch rather than K `arange` launches plus concatenation.

Clean R9700 measurements (ROCm 7.2, B=1, 64 generated tokens) are stored in
`artifacts/w4a4_measurements/`:

- context 128: 55.01 token/s, 18.18 ms/token, 4.182 GB peak allocated;
- matching HF BF16: 25.24 token/s, giving 2.18x decode throughput;
- context 4096: 39.81 token/s, 25.12 ms/token, 4.297 GB peak allocated;
- matching HF FP16 at context 4096: 19.65 token/s, giving 2.03x throughput.

The first 4K repeat is treated as a cold decode sample; the reported median is
from three deterministic repeats after one untimed prefill. Prefill is not the
current optimization target and must not be described as accelerated.

`verify_chunk` projects all speculative positions together, writes provisional
KV, and aliases the same prefix pages through B×K causal batch-decode requests.
No existing prefix KV is copied. Capacity pages are preallocated, and rejected
slots remain logically invisible until overwritten.

Before using the packed target in PARD, run the mandatory R9700 gate:

```bash
PYTHONPATH=. python -m w4a4_runtime.qualify_pard --checkpoint /path/to/checkpoint
```

It compares K=2/4/8/16 verification against sequential decoding, exercises a
page boundary, checks greedy-token parity and transaction semantics, and
reports peak VRAM. A missing GPU, native extension, tensor, or numerical match
is a hard failure.

The PARD benchmark accepts the packed target directly:

```bash
PYTHONPATH=. python -m pard_benchmark.benchmark \
  --mode pard --w4a4-checkpoint /path/to/checkpoint \
  --draft-k 12 --context-len 128 --max-new-tokens 128 \
  --compile-mode eager --local-files-only
```

PARD and PARD2 target-independent checkpoints are supported. PARD2
target-dependent mode is intentionally rejected because the packed runtime
does not expose the four target hidden-state taps required by that checkpoint.

On the R9700 context-128 / 64-token tuning workload, target verification is
1.40x, 1.52x and 2.18x faster than sequential verification for K=4, 8 and 16.
With `amd/PARD-Llama-3.2-1B`, measured steady throughput is 46.35, 74.42, 78.55
and 73.73 token/s for draft K=4, 8, 12 and 15 respectively. K=12 is therefore
the current recommendation: 1.44x faster than the measured 54.74 token/s
packed-target AR baseline, with about 6.82 GB peak allocated.

The current checkpoint is RTN. On the checked 127-token sample it records PPL
5.245 versus FP16 4.720 (+11.1%), 89.8% teacher-forced top-1 agreement, and a
16-token exact greedy prefix. GPTQ-128 calibration and full lm-eval evaluation
remain quality deliverables.
