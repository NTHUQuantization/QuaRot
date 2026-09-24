
# <img src="img/carrot.png" alt="Your Image" width="40" height="45">QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
This repository contains the code for [**QuaRot**: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456).



## Abstract
We introduce QuaRot, a new **Qua**ntization scheme based on **Rot**ations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits. QuaRot rotates LLMs in a way that removes outliers from the hidden state without changing the output, making quantization easier. This *computational invariance* is applied to the hidden state (residual) of the LLM, as well as to the activations of the feed-forward components, aspects of the attention mechanism and to the KV cache. The result is a quantized model where all matrix multiplications are performed in 4-bits, without any channels identified for retention in higher precision. Our quantized **LLaMa2-70B** model has losses of at most **0.29 WikiText perplexity** and retains **99% of the zero-shot** performance.

![Your Image](img/fig1.png)

## setup

### build 
```bash
python -m pip install -r requirements.txt

# Build fast_hadamard_transform .
python -m pip install -e third-party/hadacore --no-build-isolation -v

# Build the QuaRot C++/HIP extension .
python -m pip install -e . --no-build-isolation -v
```

# Llama 3.1 8B example

The commands below use `meta-llama/Llama-3.1-8B` and write the converted
checkpoint to `/models/quarot-llama-3.1-8b-gptq-int4`. Access to the gated
Hugging Face model is required for conversion and for the FP16 reference runs.

## Generate an INT4 checkpoint

### RTN
```bash
python e2e/checkpoint_utils/quantize_checkpoint.py \
  --model meta-llama/Llama-3.1-8B \
  --output /models/quarot-llama-3.1-8b-rtn-int4 \
  --rotation-device cuda \
  --rotation-dtype float32
```

RTN is the default quantization method. It streams safetensors checkpoints one
layer at a time. Each packed layer is written directly under `--output`; re-run
the same command to resume from those completed shards.

### GPTQ
```bash
python e2e/checkpoint_utils/quantize_checkpoint.py \
  --model meta-llama/Llama-3.1-8B \
  --output /models/quarot-llama-3.1-8b-gptq-int4 \
  --quant-method gptq \
  --cal-dataset wikitext2 \
  --nsamples 128
```

## INT4 decode benchmark

Use the Fusion harness to sweep batch sizes and context lengths with the real
converted checkpoint:

```bash
python e2e/benchmark_fusion_harness.py \
  --model /models/quarot-llama-3.1-8b-gptq-int4 \
  --batches 1 \
  --context-lengths 128,1024,2048,4096 \
  --decode-steps 16 \
  --warmup 3 \
  --repeats 5 \
  --local-files-only \
  --output /tmp/llama3_8b_fusion_benchmark.json
```

## Real INT4 versus FP16 benchmark

```bash
python e2e/benchmark_real.py \
  --int4-model /models/quarot-llama-3.1-8b-gptq-int4 \
  --fp16-model meta-llama/Llama-3.1-8B \
  --batch-size 1 \
  --prefill-seq-len 1024 \
  --decode-steps 64 \
  --output /tmp/benchmark_real_llama3_8b_gptq.json
```

Add `--int4-only` when the FP16 reference does not fit in GPU memory. This
still validates the packed checkpoint and fused dispatch, but it cannot report
an INT4-to-FP16 speedup or numerical comparison.

## Accuracy benchmark

```bash
python e2e/benchmark_accuracy.py \
  --int4-model /models/quarot-llama-3.1-8b-gptq-int4 \
  --reference-model meta-llama/Llama-3.1-8B \
  --dataset wikitext2 \
  --ppl-tokens 512 \
  --ppl-chunk 128 \
  --sequential-low-vram \
  --output /tmp/benchmark_accuracy_llama3_8b_gptq.json
```

## Fused PARD2: Qwen3-8B / 14B / 32B

This branch (`fused_pard2`) runs dense Qwen3 targets with fused W4A4KV4
HIP kernels on AMD GPUs. The supported inference contract is batch 1,
greedy decoding, thinking disabled, normal EOS handling, and `draft_k=15`.

| Target | Profile | Canonical modes | Target checkpoint |
|---|---|---|---|
| Qwen3-8B | `qwen3_8b` | AR, PARD2-TI, PARD2-TD | RTN W4A4KV4 |
| Qwen3-14B | `qwen3_14b` | AR, PARD2-TI, PARD2-TD | RTN W4A4KV4 |
| Qwen3-32B | `qwen3_32b` | AR, PARD2-TI | GPTQ W4A4KV4; RTN baseline |

TI uses the shared `amd/PARD2-Qwen3-8B` drafter for all three targets.
TD uses the matching 8B or 14B drafter. The explicit 14B-on-32B TD proxy
and its calibration/training tools are experimental, not a qualified 32B TD release.

Start with the **[Qwen3 setup and execution guide](e2e/PARD2_QWEN3.md)**
for pinned model revisions, checkpoint conversion, generation, benchmarking,
and qualification. Always pass the target, tokenizer, and draft paths explicitly
when selecting 14B or 32B; selecting a profile validates the model contract but
does not download models or replace the legacy 8B path defaults.

### Documentation

- [Source and experiment index](e2e/PARD2_SOURCE_INDEX.md): runtime, conversion, tests, and experimental tooling.
- [Integration tutorial (ZH)](e2e/PARD2_INTEGRATION_TUTORIAL_ZH.md): verifier, cache transactions, and parity debugging.
- [Optimization tutorial (ZH)](e2e/PARD2_OPTIMIZATION_TUTORIAL_ZH.md): batched LM-head, cached TD basis, and memory analysis.
- [32B experiment report (ZH)](e2e/QWEN3_32B_W4A4KV4_PARD2_REPORT_ZH.md): historical GPTQ, AR/TI qualification, and rejected TD proxy evidence.
- [8B post-merge results](pard2_post_merge_results/README.md): historical qualification and reproduction commands.

Historical reports preserve the environment and measurements of their original
runs. Their absolute paths, binary hashes, and local artifact references must be
adapted to a new machine; model weights and new raw experiment outputs are not
shipped in Git. See the setup guide for the current supported entry points.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=spcl/QuaRot&type=Date)](https://star-history.com/#spcl/QuaRot&Date)


## Citation 

The full citation is

```
@article{ashkboos2024quarot,
  title={QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs},
  author={Ashkboos, Saleh and Mohtashami, Amirkeivan and Croci, Maximilian L and Li, Bo and Jaggi, Martin and Alistarh, Dan and Hoefler, Torsten and Hensman, James},
  journal={arXiv preprint arXiv:2404.00456},
  year={2024}
}
```
