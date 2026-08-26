
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

## PARD2 speculative decoding (Qwen3-8B)

This branch integrates official PARD2-TI/TD with the fused W4A4KV4 HIP target.
The qualified configuration is batch 1, greedy decoding, normal EOS handling,
`draft_k=15`, native GQA KV4, batched LM-head, and cached TD basis.

### Documentation

- [Integration tutorial (ZH)](e2e/PARD2_INTEGRATION_TUTORIAL_ZH.md): verifier, cache transactions, parity debugging, and the first qualified runtime.
- [Optimization tutorial (ZH)](e2e/PARD2_OPTIMIZATION_TUTORIAL_ZH.md): batched LM-head, cached TD basis, memory/roofline analysis, and post-merge qualification.
- [Merge conflict record (ZH)](e2e/FUSED_V1_MERGE_CONFLICTS_ZH.md): decisions made while merging `origin/fused_v1`, rejected fallbacks, and performance evidence.
- [Post-merge result index](pard2_post_merge_results/README.md): benchmark contract, aggregate results, raw artifact names, and reproduction commands.

The older `PARD2_INTEGRATION_REPORT_ZH*.md` files are experiment journals;
the two tutorials above are the canonical onboarding documents.

### Qualified benchmark

```bash
python e2e/benchmark_pard2.py \
  --mode pard2-td \
  --dataset math_500 \
  --generated-tokens 256 \
  --warmups 8 \
  --sweeps 3 \
  --compile-mode max-autotune-no-cudagraphs \
  --precompile-draft-shapes \
  --output /tmp/pard2-td_math_500.json
```

Run AR, TI, and TD in separate processes. Then place the nine formal JSON
files under the names expected by `e2e/qualify_pard2.py`. The checked-in
qualification summary records exact parity, paired bootstrap confidence
intervals, run-level CV, and VRAM headroom.

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
