
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

# benchmark example

## synthetic benchmark

```bash
python e2e/benchmark.py \
      --batch_size 1 \
      --prefill_seq_len 2048 \
      --decode_steps 128 \
      --int4_only
```
`--int4_only` is required to run CodeLlama-34b-hf.


## Generate int4 checkpoint

### RTN
```bash
python e2e/checkpoint_utils/quantize_checkpoint.py \
  --model meta-llama/CodeLlama-34b-hf \
  --output /models/quarot-codellama-34b-rtn-int4 \
  --rotation-device cuda \
  --rotation-dtype float32
```

RTN is the default quantization method. It streams safetensors checkpoints one
layer at a time. Each packed layer is written directly under `--output`; re-run
the same command to resume from those completed shards. The dense FP16 model is
never held in host RAM.

### GPTQ
```bash
python e2e/checkpoint_utils/quantize_checkpoint.py \
  --model meta-llama/CodeLlama-34b-hf \
  --tokenizer-model hf-internal-testing/llama-tokenizer \
  --output /models/quarot-codellama-34b-gptq-int4 \
  --quant-method gptq \
  --cal-dataset wikitext2 \
  --nsamples 128
```

## Real performance benchmark

```bash
python e2e/benchmark_real.py \
  --int4-model /tmp/quarot-llama2-7b-rtn-int4 \
  --fp16-model meta-llama/Llama-2-7b-hf \
  --batch-size 1 \
  --prefill-seq-len 2048 \
  --decode-steps 128 \
  --output /tmp/benchmark_real_llama2_7b_rtn.json
```

## Accuracy benchmark

```bash
python e2e/benchmark_accuracy.py \
  --int4-model /tmp/quarot-llama2-7b-rtn-int4 \
  --reference-model meta-llama/Llama-2-7b-hf \
  --output /tmp/benchmark_accuracy_llama2_7b_rtn.json
```    

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
