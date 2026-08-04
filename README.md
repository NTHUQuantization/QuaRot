
# <img src="img/carrot.png" alt="Your Image" width="40" height="45">QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
This repository contains the code for [**QuaRot**: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456).



## Abstract
We introduce QuaRot, a new **Qua**ntization scheme based on **Rot**ations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits. QuaRot rotates LLMs in a way that removes outliers from the hidden state without changing the output, making quantization easier. This *computational invariance* is applied to the hidden state (residual) of the LLM, as well as to the activations of the feed-forward components, aspects of the attention mechanism and to the KV cache. The result is a quantized model where all matrix multiplications are performed in 4-bits, without any channels identified for retention in higher precision. Our quantized **LLaMa2-70B** model has losses of at most **0.29 WikiText perplexity** and retains **99% of the zero-shot** performance.

![Your Image](img/fig1.png)

### Clone and install

```bash
git clone https://github.com/NTHUQuantization/QuaRot.git
```

download composable kernel
```bash
cd ~/QuaRot/third-party
git clone --depth 1 --filter=blob:none --sparse \
  --branch develop \
  https://github.com/ROCm/rocm-libraries.git \
  rocm-libraries

cd rocm-libraries

git sparse-checkout set projects/composablekernel
```
download hadmard transform

build 
```bash
python -m pip install -r requirements.txt

# Build fast_hadamard_transform .
python -m pip install -e third-party/hadacore --no-build-isolation -v

# Build the QuaRot C++/HIP extension .
python -m pip install -e . --no-build-isolation -v
```

## Running benchmark

```bash
python e2e/benchmark.py \
    --batch_size 1 \
    --prefill_seq_len 512 \
    --decode_steps 128
```

For simulation-only experiments, see the [fake_quant](fake_quant) directory.

### Star History

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
