# QuaRot HIP Kernel Fusion 評估報告

日期：2026-07-04  
環境：ROCm container `rocm/vllm-dev:rocm7.2_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.14.0rc0`  
GPU：`gfx1201`，64 CU，wavefront size 32  
Profiling 工具：`rocprofv3`

## 摘要

本報告合併兩層實驗：

1. **Single decoder layer decode-step 評估**  
   比原本 projection 後的 kernel-fusion ablation 更接近真實 LLM decode layer。Harness 包含 RMSNorm/residual、QKV projection、RoPE、KV append、attention decode、O projection、FFN gate/up/down projection；fused 版本只替換 K1/K2/K3/FFN path，其餘 layer 計算保持相同。

2. **元件級 fused kernel profiling**  
   分別量測 FFN、K1、K2、K3 的 unfused baseline、existing path、fused prototype，用來確認每個 kernel fusion 本身是否有效。

最重要的結論是：**fused QuaRot 對 QuaRot original unfused path 有穩定 decoder-layer speedup，但相對 FP16 baseline 的改善有限且 shape-dependent。** 因此目前不能宣稱 full LLM end-to-end 有 100x speedup；本報告結果應解讀為 **single decoder layer / projected kernel path speedup**。

## 實驗產物

主要腳本與結果：

- `single_decoder_layer_benchmark.py`：single decoder layer correctness、latency、ablation harness。
- `single_decoder_layer_results/correctness.csv`：完整 correctness 結果。
- `single_decoder_layer_results/latency.csv`：完整 latency 結果。
- `single_decoder_layer_results/ablation.csv`：完整 ablation 結果。
- `rocprof_single_decoder_layer/summary.md`：single decoder layer rocprofv3 kernel breakdown。
- `profiler_workloads.py`：元件級 kernel profiling workload。
- `rocprof_results_kernel_fusion_raw/summary.md`：元件級 profiling summary。
- `single_decoder_layer_report_zh.md`：single decoder layer 詳細報告來源。

## Single Decoder Layer Harness

### Benchmark 設定

- Decode-step 設定：前 `L-1` token KV cache 已 prefilled；timed region 只測 current token decode layer。
- Timed region 包含：
  - RMSNorm
  - QKV projection
  - RoPE
  - KV append
  - attention decode
  - O projection + residual
  - FFN gate/up/down projection
- 三條主路徑：
  - `fp16_baseline`：FP16 KV cache + FP16 attention/FFN。
  - `quarot_unfused`：INT4/rotation path，但 Hadamard、quant、dequant、decode 分開做。
  - `fused_quarot`：使用目前 HIP fused K1/K2/K3/FFN kernels。
- Ablation：
  - `quarot_unfused`
  - `k1_fused`
  - `k1_k2_fused`
  - `attention_fused`
  - `fused_quarot`
- Shapes：
  - batch = `1,2,4,8`
  - context length L = `10,128,1024,4096`
  - FFN hidden = `11008,14336`
- Timing：HIP/PyTorch CUDA event，`warmup=2`、`iters=5`。
- 權重/輸入：固定 random seed；三條路徑共用同一組 current hidden、cache state 與 layer weights。

### Correctness

下表為 fused QuaRot output hidden states 相對 QuaRot original unfused 的差異。Relative error 容易被接近 0 的 hidden element 放大，因此主要以 max/mean error 判讀。

代表表：batch=1、FFN hidden=14336。

| batch | seq_len | ffn_hidden | max_error | mean_error | mean_relative_error | tolerance | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 10 | 14336 | 1.168 | 0.232 | 0.677 | max<=1.25,mean<=0.25 | PASS |
| 1 | 128 | 14336 | 0.615 | 0.123 | 0.518 | max<=1.25,mean<=0.25 | PASS |
| 1 | 1024 | 14336 | 0.180 | 0.038 | 11.897 | max<=1.25,mean<=0.25 | PASS |
| 1 | 4096 | 14336 | 0.119 | 0.026 | 13.297 | max<=1.25,mean<=0.25 | PASS |

完整結果在 `single_decoder_layer_results/correctness.csv`。

### Decoder Layer Latency

全 shape 平均：

- fused QuaRot 相對 QuaRot unfused：`6.36x`
- 範圍：`3.81x` 到 `21.16x`
- fused QuaRot 相對 FP16 baseline：`1.21x`
- 範圍：`0.74x` 到 `2.29x`

代表表：batch=1、FFN hidden=14336。

| batch | seq_len | ffn_hidden | FP16 baseline (ms) | QuaRot unfused (ms) | Fused QuaRot (ms) | fused vs unfused | fused vs FP16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 14336 | 1.174 | 4.348 | 1.077 | 4.038x | 1.090x |
| 1 | 128 | 14336 | 1.149 | 4.242 | 1.082 | 3.919x | 1.062x |
| 1 | 1024 | 14336 | 1.235 | 4.641 | 1.126 | 4.123x | 1.097x |
| 1 | 4096 | 14336 | 1.530 | 5.639 | 1.300 | 4.339x | 1.177x |

完整結果在 `single_decoder_layer_results/latency.csv`。

### Decoder Layer Ablation

代表表：batch=1、FFN hidden=14336。

| seq_len | variant | latency_ms | speedup_vs_unfused |
| ---: | --- | ---: | ---: |
| 10 | quarot_unfused | 5.205 | 1.000x |
| 10 | k1_fused | 2.825 | 1.842x |
| 10 | k1_k2_fused | 2.353 | 2.212x |
| 10 | attention_fused | 1.639 | 3.176x |
| 10 | fused_quarot | 1.140 | 4.566x |
| 128 | quarot_unfused | 5.538 | 1.000x |
| 128 | k1_fused | 5.543 | 0.999x |
| 128 | k1_k2_fused | 2.783 | 1.990x |
| 128 | attention_fused | 1.639 | 3.379x |
| 128 | fused_quarot | 1.080 | 5.126x |
| 1024 | quarot_unfused | 5.145 | 1.000x |
| 1024 | k1_fused | 2.523 | 2.039x |
| 1024 | k1_k2_fused | 2.523 | 2.040x |
| 1024 | attention_fused | 1.629 | 3.157x |
| 1024 | fused_quarot | 1.130 | 4.552x |
| 4096 | quarot_unfused | 5.796 | 1.000x |
| 4096 | k1_fused | 4.871 | 1.190x |
| 4096 | k1_k2_fused | 2.465 | 2.351x |
| 4096 | attention_fused | 1.775 | 3.265x |
| 4096 | fused_quarot | 1.292 | 4.487x |

完整結果在 `single_decoder_layer_results/ablation.csv`。

### Single Decoder Layer rocprofv3

代表點：batch=1、L=128、FFN hidden=14336、iters=5。  
rocprofv3 執行期間出現 timestamp swap warnings；CSV stats 已產生，但細粒度 kernel duration 應視為 profiling 近似值。

| variant | kernel_calls_per_iter | kernel_time_us_per_iter |
| --- | ---: | ---: |
| fp16_baseline | 97.800 | 1586.151 |
| fused_quarot | 94.800 | 1599.432 |
| quarot_unfused | 390.800 | 2582.754 |

觀察：

- QuaRot unfused path 的 kernel calls 明顯較高，主要來自 PyTorch separate Hadamard/quant/dequant/copy kernels。
- Fused QuaRot 大幅降低 QuaRot unfused 的 kernel calls，但完整 decoder layer 中 GEMM/projection 仍是主要時間來源。
- Fused QuaRot 與 FP16 baseline 的 kernel time 接近，代表目前融合主要解決 QuaRot unfused overhead，而不是取代 GEMM 主成本。

## 元件級 Kernel Fusion Profiling

來源：`rocprof_results_kernel_fusion_raw/summary.md`。  
元件級測試的重點是確認各 fused kernel 本身是否比對應 unfused baseline 更有效。

| Block | Variant | Kernel time / iter (us) | Kernel calls / iter | Min semantic IO (KiB) | SQ waves / iter |
| --- | --- | ---: | ---: | ---: | ---: |
| FFN | unfused PyTorch | 220.875 | 67.00 | n/a | 4120.000 |
| FFN | hadacore pipeline | 140.596 | 22.00 | n/a | 1656.000 |
| FFN | fused prototype | 3.490 | 1.00 | 63.109 | 448.000 |
| K1 | unfused PyTorch, RoPE off | 437.843 | 134.55 | n/a | 918.200 |
| K1 | unfused PyTorch, RoPE on | 545.682 | 151.55 | n/a | 1038.200 |
| K1 | fused append, RoPE off | 5.331 | 1.00 | 5.062 | 32.000 |
| K1 | fused append, RoPE on | 9.603 | 1.00 | 5.062 | 32.000 |
| K2 | QuaRot original unfused INT4 dequant + FP16 decode | 988.947 | 23.90 | n/a | 378812.200 |
| K2 | baseline FP16 KV decode | 383.278 | 1.00 | 16384.000 | 32.000 |
| K2 | optimized INT4 KV decode | 234.357 | 1.00 | 4352.000 | 32.000 |
| K3 | unfused PyTorch | 251.394 | 64.05 | n/a | 696.800 |
| K3 | fused output quant | 3.412 | 1.00 | 10.031 | 128.000 |

### FFN

FFN fused prototype 合併：

`SiLU(gate) * up -> 256-d normalized Hadamard -> amax -> scale=max/7 -> signed INT4 pack`

以 `H_int=14336`、decode `B*T=1` 測試：

- PyTorch unfused：`220.875 us`，約 `67` kernels/iter。
- hadacore pipeline：`140.596 us`，約 `22` kernels/iter。
- fused prototype：`3.490 us`，`1` kernel/iter。

主要收益來自移除 PyTorch 中間 tensor、減少 kernel launch，並把 activation/Hadamard/quant/pack 合併在同一個 block 內完成。

### K1: Append KV + Hadamard + Quant

正式 unfused baseline 用 PyTorch 做 RoPE、Hadamard、dynamic quant、pack，最後寫入 paged KV cache。

- RoPE off：`437.843 us -> 5.331 us`
- RoPE on：`545.682 us -> 9.603 us`
- kernel calls 從約 `135-152` 降到 `1`

K1 fusion 方向有效，但整合時需特別注意 paged KV cache layout、page index、last page offset、RoPE position 與 K2 decode 的格式一致性。

### K2: QuaRot Original Unfused vs INT4 Decode

K2 有三個對照：

- baseline FP16 KV decode：KV cache 直接用 FP16，呼叫 FlashInfer FP16 decode。
- QuaRot original unfused INT4 dequant + FP16 decode：KV cache 是 INT4 packed + scale，但先用獨立 PyTorch kernels unpack/dequant 回 FP16 paged KV cache，再呼叫 FlashInfer FP16 decode。
- optimized INT4 KV decode：直接呼叫 FlashInfer INT4 decode，在 decode kernel 內讀 packed INT4 與 scale 並完成 dequant/attention。

結果：

- baseline FP16 KV decode：`383.278 us`
- QuaRot original unfused INT4 dequant + FP16 decode：`988.947 us`
- optimized INT4 KV decode：`234.357 us`
- optimized INT4 decode 相對 QuaRot original unfused 約 `4.22x` faster
- optimized INT4 decode 相對 FP16 KV decode 約 `1.64x` faster
- 理論 KV/cache 讀取量：FP16 約 `16384 KiB`，INT4+scale 約 `4352 KiB`

### K3: Attention Output Hadamard + Quant

K3 baseline 用 PyTorch 做 `[1,4096] -> [1,16,256]` block Hadamard，再做 grouped quant/pack。

- PyTorch unfused：`251.394 us`，約 `64` kernels/iter。
- fused prototype：`3.412 us`，`1` kernel/iter。
- fused kernel 最小語意 I/O 約 `10.031 KiB`。

K3 保持獨立 kernel 是合理的；它位於 attention output 後處理/下一層前處理之間，不建議硬塞進 FlashAttention。

## Counter 與 Memory Traffic 狀態

`rocprofv3 --list-avail` 在 `gfx1201` 上列得出 memory/LDS/occupancy 相關 counter，包括：

- Memory：`FetchSize`、`GL2C_EA_RDREQ_{32,64,128}B_sum`、`GL2C_EA_WRREQ_64B_sum`
- LDS：`SQ_INSTS_LDS`、`SQC_LDS_IDX_ACTIVE`、`SQC_LDS_BANK_CONFLICT`
- Activity / waves：`SQ_WAVES_sum`、`GRBM_GUI_ACTIVE`、`GRBM_COUNT`
- Occupancy：`MeanOccupancyPerCU`

實測結果：

- `SQ_WAVES_sum`、`GRBM_GUI_ACTIVE`、`GRBM_COUNT`、`CU_NUM`、`SIMD_NUM` 有非零值。
- GL2C memory counters、LDS counters、`SQ_WAVE_CYCLES`、`MeanOccupancyPerCU` 在本環境回傳 0。
- `rocprof-compute` 可啟動，但 ROCm 7.2 版本支援清單只有 `gfx908/gfx90a/gfx940/gfx941/gfx942/gfx950`，本機 `gfx1201` 不支援，無法產官方 roofline。

因此本報告使用：

- rocprofv3 kernel time
- kernel calls
- SQ waves / GRBM activity
- 理論最小 semantic I/O

作為 memory traffic 補充，而不直接解讀 GL2C/LDS/occupancy counter。

代表點 B=1、L=128、FFN hidden=14336 的最低資料量估算：

| component | formula | bytes |
| --- | --- | ---: |
| K2 FP16 KV cache read | B * L * D * 2(K,V) * 2 bytes | 2097152 |
| K2 INT4 KV cache read | B * L * D * 2(K,V) * 0.5 bytes + scales | 557056 |
| K1 current read/write | read FP16 K/V, write INT4 K/V + scale | 20736 |
| K3 output quant | read FP16 4096, write INT4 4096 + 16 scales | 10272 |
| FFN intermediate quant | read FP16 gate/up, write INT4 hidden + scales | 64624 |

以上 bytes 未包含 GEMM 讀權重、cache miss、alignment、temporary tensor、PyTorch dispatcher overhead。

## API 與 Correctness Test

K1/K3 已補上一層穩定 Python-facing API，位置在 `attention_fusion/quarot_attention_fusion.py`：

- `make_uniform_paged_kv_metadata(...)`
- `allocate_quantized_kv_cache(...)`
- `append_quantized_kv_decode(...)`
- `quantize_attention_output(...)`
- `*_reference(...)`

固定 correctness test：`attention_fusion/test_attention_fusion_correctness.py`。

測試內容：

- K1 RoPE off/on 的 packed mismatch rate <= `1%`
- K1 scale max error <= `0.006`
- K3 packed mismatch rate <= `1%`
- K3 scale max error <= `0.001`

已通過：

```bash
cd /workspace/QuaRot/attention_fusion
python3 test_attention_fusion_correctness.py --k3-rows 1,4,16
python3 test_attention_fusion_correctness.py --batch 2 --heads 8 --seq-len 257 --page-size 128 --k3-rows 2,8
```

Single decoder layer correctness 則固定輸出 hidden-state max/mean/relative error，並用 `max_error <= 1.25` 且 `mean_error <= 0.25` 判定 fused QuaRot 相對 QuaRot unfused 是否通過。

## Integration 狀態

目前 repo 沒有完整 HuggingFace/Llama model forward；可整合的實體是 projection 後的 decode kernel path，以及本次新增的 synthetic single decoder layer harness。

已完成：

- `quarot_model_integration.py`：projection 後 decode-layer wrapper。
- `test_quarot_model_integration.py`：batch=1/2、RoPE on/off、FFN hidden=14336/11008 smoke/integration test。
- `single_decoder_layer_benchmark.py`：包含 norm/projection/residual/FFN projection 的 single decoder layer harness。

`QuaRotFusedDecodeLayer` 串接：

- K1：`append_quantized_kv_decode`
- K2：`flashinfer_hip.batch_decode_i4`
- K3：`quantize_attention_output`
- FFN：`ffn_fusion_hip.fused_ffn_silu_hadamard_quant`

已通過：

```bash
cd /workspace/QuaRot
python3 test_quarot_model_integration.py
python3 test_quarot_model_integration.py --rope
python3 test_quarot_model_integration.py --batch 2 --ffn-hidden 11008
python3 test_quarot_model_integration.py --batch 2 --ffn-hidden 11008 --rope
```

## 舊 Projection-Only Ablation 的定位

早期 `profile_quarot_model_integration.py` 使用 projection 後 tensors 做 ablation，曾量到 `full_fused` 相對 `unfused_all` 約 `100x` event speedup。這個結果只代表 **projection 後 kernel path**，不包含 RMSNorm、QKV/O/FFN linear projection、residual 等 decoder layer 成本。

因此新版報告以 single decoder layer harness 作為主要結論；projection-only ablation 僅保留為 kernel path 上限參考，不作為 full model/end-to-end speedup 宣稱。

## 目前缺少項目

1. **完整 Llama/QuaRot model class 接線**  
   本 repo 內仍沒有完整 HF/Llama model forward、真實 checkpoint weights、tokenizer、sampling loop。若要做真正 full model end-to-end，需要接到外部 model repo。

2. **真實模型輸出品質驗證**  
   目前 correctness 是 random weights/cache 的 hidden-state 差異與 kernel-level mismatch；尚未驗證真實 logits、perplexity 或 task accuracy。

3. **更長時間 profiling 與多次重跑統計**  
   Single decoder layer grid 目前使用 `warmup=2`、`iters=5`。正式發表或 demo 前建議把代表 shapes 拉高到更多 iterations，並回報 variance。

4. **硬體 counter 限制**  
   `gfx1201` 上 GL2C/LDS/occupancy counter 仍無法取得有效數值，只能以 kernel time/calls/SQ waves 與理論 traffic 補充。

## Kernel Fusion 可行性評估

目前 kernel fusion 方向可行，而且在 QuaRot original unfused path 上收益明確。主要風險在模型整合邊界，而不是 fused kernel 本身。

- FFN fused kernel 很適合繼續優化，資料流線性、group size 固定、輸出即 packed INT4 + fp16 scale。
- K1 需要持續驗證 paged KV cache layout、page index、last page offset、RoPE position。
- K2 已有可用 INT4 decode kernel，短期重點是確保 K1 產生的 INT4 KV cache layout 完全符合 K2。
- K3 保持獨立 kernel 較安全，後續只需確認下一個 block/linear layer 是否吃 packed INT4 + scale。

## 結論

1. 元件級 profiling 顯示 FFN/K1/K2/K3 fusion 都能顯著降低 PyTorch unfused path 的 kernel calls 與 kernel time。
2. Single decoder layer harness 顯示 fused QuaRot 相對 QuaRot original unfused 平均約 `6.36x` speedup。
3. 相對 FP16 baseline 平均約 `1.21x`，且 shape-dependent；這說明 GEMM/projection 仍主導完整 decoder layer 成本。
4. 目前結果不能外推成 full LLM end-to-end 100x speedup。較精準的說法是：**目前 fused kernels 可有效加速 QuaRot decode layer 中 rotation/quant/dequant/INT4 decode 相關 path，但整體模型速度仍取決於 GEMM、projection、runtime scheduling 與完整模型整合。**

## Reproduce

Single decoder layer：

```bash
cd /workspace/QuaRot
python3 single_decoder_layer_benchmark.py \
  --mode bench \
  --batches 1,2,4,8 \
  --lengths 10,128,1024,4096 \
  --ffn-hidden 11008,14336 \
  --iters 5 \
  --warmup 2 \
  --out-dir single_decoder_layer_results

python3 single_decoder_layer_benchmark.py \
  --mode ablation \
  --batches 1,2,4,8 \
  --lengths 10,128,1024,4096 \
  --ffn-hidden 11008,14336 \
  --iters 5 \
  --warmup 2 \
  --out-dir single_decoder_layer_results
```

rocprofv3 代表點：

```bash
cd /workspace/QuaRot
mkdir -p rocprof_single_decoder_layer
for v in fp16_baseline quarot_unfused fused_quarot; do
  rocprofv3 --kernel-trace --stats -f csv \
    -d rocprof_single_decoder_layer \
    -o ${v}_B1_L128_H14336 \
    -- python3 single_decoder_layer_benchmark.py \
      --mode variant \
      --variant ${v} \
      --batch 1 \
      --seq-len 128 \
      --ffn-hidden-single 14336 \
      --iters 5
done

python3 summarize_single_decoder_rocprof.py \
  --dir rocprof_single_decoder_layer \
  --iters 5 \
  --top 10
```

元件級 profiling：

```bash
cd /workspace/QuaRot
python3 profiler_workloads.py --help
python3 summarize_rocprof_kernel_fusion.py --help
```
