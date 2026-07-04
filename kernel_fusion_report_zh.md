# QuaRot HIP Kernel Fusion 評估報告

日期：2026-07-04  
環境：ROCm container `rocm/vllm-dev:rocm7.2_navi_ubuntu22.04_py3.10_pytorch_2.9_vllm_0.14.0rc0`  
GPU：`gfx1201`，64 CU，wavefront size 32  
Profiling 工具：`rocprofv3`

## 摘要

本報告合併三層實驗：

1. **Llama-3.1 8B formal token-by-token 評估**  
   使用真實 `meta-llama/Llama-3.1-8B` 權重與 tokenizer。Prefill 使用 HF model，decode 經由 `QuaRotLlamaForCausalLM` wrapper 的 `prefill()`、`decode_one()`、`generate()` 執行，可比較 FP16、QuaRot original unfused、Fused QuaRot。

2. **Single decoder layer decode-step 評估**  
   比原本 projection 後的 kernel-fusion ablation 更接近真實 LLM decode layer。Harness 包含 RMSNorm/residual、QKV projection、RoPE、KV append、attention decode、O projection、FFN gate/up/down projection；fused 版本只替換 K1/K2/K3/FFN path，其餘 layer 計算保持相同。

3. **元件級 fused kernel profiling**  
   分別量測 FFN、K1、K2、K3 的 unfused baseline、existing path、fused prototype，用來確認每個 kernel fusion 本身是否有效。

最重要的結論是：**fused QuaRot 對 QuaRot original unfused INT4 path 有穩定 decoder-layer speedup，但 FP16 latency 仍需作為參考，因為 GEMM/projection 仍主導完整模型成本。** 因此目前不能宣稱 full LLM end-to-end 有 100x speedup；本報告結果應解讀為 **formal token-by-token decode / decoder layer / projected kernel path speedup**。

本報告所有 speedup 倍數統一使用：

```text
speedup_vs_unfused_INT4 = latency(unfused_INT4) / latency(variant)
```

其中 `unfused_INT4` 指 QuaRot original unfused path：Hadamard、dynamic quant/pack、dequant、INT4 decode 相關步驟以獨立 PyTorch/既有 kernels 串接。FP16 欄位只作為參考 latency；若表中列出 FP16 的 speedup，也同樣是相對 unfused_INT4，而不是以 FP16 當 baseline。

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
- `llama31_quarot/`：Llama-3.1 8B full-model baseline、quality、profiling harness。
- `llama31_full_model_plan_zh.md`：Llama-3.1 8B 接線與驗證計畫。
- `llama31_full_model_results_fp16_B1/`：Llama-3.1 8B FP16 B=1 longer-run baseline。
- `llama31_full_model_results/`：Llama-3.1 8B FP16 quality self-check。
- `llama31_full_model_rocprof/`：Llama-3.1 8B FP16 rocprofv3 representative trace。
- `llama31_decode_path_results/`：Llama-3.1 8B 真實權重 32 層 decode-path B=1/2/4, L=10/128/1024 latency/correctness。
- `llama31_decode_path_results_L4096/`：Llama-3.1 8B decode-path B=1, L=4096 補充長 context 結果。
- `llama31_decode_path_rocprof/`：Llama-3.1 8B decode-path rocprofv3 representative trace。
- `llama31_decode_path_report_zh.md`：Llama-3.1 8B decode-path 中文報告。
- `llama31_formal_latency_fp16/`：formal token-by-token FP16 baseline latency grid。
- `llama31_formal_latency_quarot_unfused/`：formal token-by-token QuaRot original unfused latency grid。
- `llama31_formal_latency_fused_quarot/`：formal token-by-token fused QuaRot latency grid。
- `llama31_formal_quality_fp16_vs_unfused/`：FP16 vs QuaRot unfused logits/generation quality。
- `llama31_formal_quality_fp16_vs_fused/`：FP16 vs fused QuaRot logits/generation quality。
- `llama31_formal_quality_unfused_vs_fused/`：QuaRot unfused vs fused QuaRot logits/generation quality。
- `llama31_formal_rocprof/`：formal full-model representative rocprofv3 traces。
- `llama31_formal_full_model_report_zh.md`：formal token-by-token full-model 中文報告。
- `flashinfer_gqa_results/`：GQA-aware K2 decode correctness/latency。
- `rocprof_flashinfer_gqa/`：GQA-aware K2 decode rocprofv3 representative trace。

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

- fused QuaRot 相對 unfused_INT4：`6.36x`
- 範圍：`3.81x` 到 `21.16x`
- FP16 參考路徑相對 unfused_INT4：`5.17x`
- 範圍：`2.05x` 到 `12.51x`

代表表：batch=1、FFN hidden=14336。

| batch | seq_len | ffn_hidden | unfused_INT4 ms | FP16 ms | Fused QuaRot ms | FP16 speedup_vs_unfused_INT4 | Fused speedup_vs_unfused_INT4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 14336 | 4.348 | 1.174 | 1.077 | 3.704x | 4.038x |
| 1 | 128 | 14336 | 4.242 | 1.149 | 1.082 | 3.691x | 3.919x |
| 1 | 1024 | 14336 | 4.641 | 1.235 | 1.126 | 3.758x | 4.123x |
| 1 | 4096 | 14336 | 5.639 | 1.530 | 1.300 | 3.686x | 4.339x |

完整結果在 `single_decoder_layer_results/latency.csv`。

### Decoder Layer Ablation

代表表：batch=1、FFN hidden=14336。

| seq_len | variant | latency_ms | speedup_vs_unfused_INT4 |
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
| quarot_unfused | 390.800 | 2582.754 |
| fp16_baseline | 97.800 | 1586.151 |
| fused_quarot | 94.800 | 1599.432 |

觀察：

- QuaRot unfused path 的 kernel calls 明顯較高，主要來自 PyTorch separate Hadamard/quant/dequant/copy kernels。
- Fused QuaRot 相對 unfused_INT4 的 kernel time speedup 約 `1.61x`，kernel calls 從 `390.8` 降到 `94.8`。
- FP16 參考路徑相對 unfused_INT4 的 kernel time speedup 約 `1.63x`。Fused QuaRot 與 FP16 參考接近，代表目前融合主要解決 QuaRot unfused overhead，而不是取代 GEMM 主成本。

## 元件級 Kernel Fusion Profiling

來源：`rocprof_results_kernel_fusion_raw/summary.md`。  
元件級測試的重點是確認各 fused kernel 本身是否比對應 unfused_INT4 baseline 更有效。

| Block | Variant | Kernel time / iter (us) | speedup_vs_unfused_INT4 | Kernel calls / iter | Min semantic IO (KiB) | SQ waves / iter |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FFN | unfused_INT4 PyTorch | 220.875 | 1.00x | 67.00 | n/a | 4120.000 |
| FFN | hadacore pipeline | 140.596 | 1.57x | 22.00 | n/a | 1656.000 |
| FFN | fused prototype | 3.490 | 63.29x | 1.00 | 63.109 | 448.000 |
| K1 RoPE off | unfused_INT4 PyTorch | 437.843 | 1.00x | 134.55 | n/a | 918.200 |
| K1 RoPE off | fused append | 5.331 | 82.13x | 1.00 | 5.062 | 32.000 |
| K1 RoPE on | unfused_INT4 PyTorch | 545.682 | 1.00x | 151.55 | n/a | 1038.200 |
| K1 RoPE on | fused append | 9.603 | 56.82x | 1.00 | 5.062 | 32.000 |
| K2 | unfused_INT4 dequant + FP16 decode | 988.947 | 1.00x | 23.90 | n/a | 378812.200 |
| K2 | FP16 KV decode reference | 383.278 | 2.58x | 1.00 | 16384.000 | 32.000 |
| K2 | optimized INT4 KV decode | 234.357 | 4.22x | 1.00 | 4352.000 | 32.000 |
| K3 | unfused_INT4 PyTorch | 251.394 | 1.00x | 64.05 | n/a | 696.800 |
| K3 | fused output quant | 3.412 | 73.68x | 1.00 | 10.031 | 128.000 |

### FFN

FFN fused prototype 合併：

`SiLU(gate) * up -> 256-d normalized Hadamard -> amax -> scale=max/7 -> signed INT4 pack`

以 `H_int=14336`、decode `B*T=1` 測試：

- PyTorch unfused_INT4：`220.875 us`，約 `67` kernels/iter，`1.00x`。
- hadacore pipeline：`140.596 us`，約 `22` kernels/iter，`1.57x` vs unfused_INT4。
- fused prototype：`3.490 us`，`1` kernel/iter，`63.29x` vs unfused_INT4。

主要收益來自移除 PyTorch 中間 tensor、減少 kernel launch，並把 activation/Hadamard/quant/pack 合併在同一個 block 內完成。

### K1: Append KV + Hadamard + Quant

正式 unfused baseline 用 PyTorch 做 RoPE、Hadamard、dynamic quant、pack，最後寫入 paged KV cache。

- RoPE off：`437.843 us -> 5.331 us`，`82.13x` vs unfused_INT4
- RoPE on：`545.682 us -> 9.603 us`，`56.82x` vs unfused_INT4
- kernel calls 從約 `135-152` 降到 `1`

K1 fusion 方向有效，但整合時需特別注意 paged KV cache layout、page index、last page offset、RoPE position 與 K2 decode 的格式一致性。

### K2: QuaRot Original Unfused vs INT4 Decode

K2 有三個對照：

- FP16 KV decode reference：KV cache 直接用 FP16，呼叫 FlashInfer FP16 decode。
- QuaRot original unfused INT4 dequant + FP16 decode：KV cache 是 INT4 packed + scale，但先用獨立 PyTorch kernels unpack/dequant 回 FP16 paged KV cache，再呼叫 FlashInfer FP16 decode。
- optimized INT4 KV decode：直接呼叫 FlashInfer INT4 decode，在 decode kernel 內讀 packed INT4 與 scale 並完成 dequant/attention。

結果：

- QuaRot original unfused INT4 dequant + FP16 decode：`988.947 us`，`1.00x`
- FP16 KV decode reference：`383.278 us`，`2.58x` vs unfused_INT4
- optimized INT4 KV decode：`234.357 us`，`4.22x` vs unfused_INT4
- 理論 KV/cache 讀取量：FP16 約 `16384 KiB`，INT4+scale 約 `4352 KiB`

### K3: Attention Output Hadamard + Quant

K3 baseline 用 PyTorch 做 `[1,4096] -> [1,16,256]` block Hadamard，再做 grouped quant/pack。

- PyTorch unfused_INT4：`251.394 us`，約 `64` kernels/iter，`1.00x`。
- fused prototype：`3.412 us`，`1` kernel/iter，`73.68x` vs unfused_INT4。
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

## Llama-3.1 8B Full Model 計畫

使用模型：`meta-llama/Llama-3.1-8B`。

已新增 `llama31_quarot/`，其中 `fp16_hf` mode 是實際 HuggingFace full-model baseline：

- `benchmark_full_model.py`：prefill/decode latency，支援 warmup、iters、repeats、p50/p90/p95。
- `validate_quality.py`：logits top-k、KL divergence、greedy generation sample。
- `decode_paths.py`：使用真實 Llama 權重與 32 層 decoder 的 single-token decode-path harness。
- `hf_quarot_model.py`：formal token-by-token wrapper，提供 `prefill()`、`decode_one()`、`generate()`。
- `profile_full_model_rocprof.sh`：rocprofv3 representative profiling。
- `summarize_full_model_results.py`：輸出 full-model 中文 summary。
- `summarize_decode_paths.py`：輸出 decode-path 中文 summary。
- `summarize_formal_full_model.py`：輸出 formal full-model 中文 summary。
- `model_patch.py`：檢查目前 fused kernels 是否可直接接上 Llama-3.1 8B。

目前已使用 `HF_TOKEN` 成功載入 `meta-llama/Llama-3.1-8B`，並完成 FP16 / QuaRot unfused / Fused QuaRot 的 formal latency grid、logits/generation quality 與 rocprofv3 representative trace。未設定 token 時，腳本會 fail fast：

```text
HF_TOKEN is required for gated Meta Llama repositories.
```

### GQA-Aware K1/K2 狀態

Llama-3.1 8B 使用 GQA：

- Q heads = `32`
- KV heads = `8`
- head_dim = `128`
- hidden size = `4096`
- FFN hidden = `14336`

目前已完成 GQA-aware K1/K2 kernel support：

1. K1 append 可用 `num_heads=kv_heads=8` 寫入 INT4 paged KV cache。
2. K2 新增 `batch_decode_f16_gqa` / `batch_decode_i4_gqa`。
3. GQA decode kernel 使用 `grid.y = num_q_heads`，KV load 使用 `kv_head = q_head / (num_q_heads / num_kv_heads)`。
4. paged KV cache 實際只存 `kv_heads=8`，不需要把 KV repeat 成 32 heads。

K1 GQA shape correctness：

- command：`test_attention_fusion_correctness.py --batch 2 --heads 8 --seq-len 257 --page-size 128 --k3-rows 2,8`
- result：PASS
- K1 mismatch rate <= `0.004883`
- K1 scale max error = `0.00390625`

K2 GQA correctness/latency：

- q_heads = `32`
- kv_heads = `8`
- reference：把 KV repeat 成 32 heads 後跑舊 non-GQA decode。
- result：f16/i4 全 shape max error = `0`，mean error = `0`。

代表表：

| batch | L | dtype | GQA latency (ms) | expanded-reference latency (ms) | GQA / expanded latency ratio |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 128 | f16 | 0.0178 | 0.0166 | 1.072 |
| 1 | 128 | i4 | 0.0170 | 0.0163 | 1.043 |
| 1 | 4096 | f16 | 0.3281 | 0.3315 | 0.990 |
| 1 | 4096 | i4 | 0.2005 | 0.2024 | 0.991 |
| 8 | 128 | f16 | 0.0244 | 0.0276 | 0.884 |
| 8 | 128 | i4 | 0.0194 | 0.0194 | 1.000 |
| 8 | 4096 | f16 | 0.8302 | 1.1520 | 0.721 |
| 8 | 4096 | i4 | 0.2817 | 0.3113 | 0.905 |

完整結果在 `flashinfer_gqa_results/gqa_decode.csv`。rocprof 代表點在 `rocprof_flashinfer_gqa/summary.md`，可看到 `BatchDecodeWithPagedKVGQAKernel` 已被 dispatch。

因此目前 `quarot_unfused` / `fused_quarot` full-model modes 不再卡於 GQA kernel 本身；已可透過 formal token-by-token wrapper 執行 `prefill()`、`decode_one()`、`generate()`。剩下未做的是直接 monkey-patch HuggingFace `LlamaDecoderLayer` / native `transformers.generate()` cache replacement。

詳細計畫見 `llama31_full_model_plan_zh.md`。

### Llama-3.1 8B Formal Token-by-Token Harness

`llama31_quarot/hf_quarot_model.py` 使用真實 `meta-llama/Llama-3.1-8B` 權重，新增 `QuaRotLlamaForCausalLM` wrapper：

- `prefill()`：使用 HF model 產生 prompt KV cache，並轉成 FP16 paged cache 或 QuaRot INT4 paged cache。
- `decode_one()`：正式 token-by-token decode API，逐層保留 HF RMSNorm、QKV/O projection、residual、FFN gate/up/down projection、final norm/lm head，只替換 QuaRot K1/K2/K3/FFN path。
- `generate()`：wrapper 內的 greedy generation loop，可跑 `fp16_hf`、`quarot_unfused`、`fused_quarot`。

這不是直接 patch Transformers internals；native `transformers.generate()` / `LlamaDecoderLayer` cache replacement 仍是下一步工程整合。

三條比較路徑：

- `fp16_hf`：HF prefill + FP16 paged KV + GQA-aware FP16 decode。
- `quarot_unfused`：PyTorch Hadamard/INT4 pack/dequant + GQA-aware FP16 decode。
- `fused_quarot`：HIP K1/K2/K3/FFN fused kernels。

Correctness：

- `fp16_hf` vs `quarot_unfused` / `fused_quarot`：top1/top10 目前為 `0`，max error 約 `13-22`，mean error 約 `2`。這是預期限制，因為尚未完成正式 QuaRot 權重旋轉與 calibration。
- `quarot_unfused` vs `fused_quarot`：top1 match 兩個 prompt 為 `1.0`、一個 prompt 為 `0.0`；top10 overlap 約 `0.8-0.9`，mean error 約 `0.25-0.28`。
- generation samples 已輸出到 `llama31_formal_quality_* /generation_samples.md`。

完整 decode latency grid：

| batch | context_len | unfused_INT4 ms/token | FP16 ms/token | Fused QuaRot ms/token | FP16 speedup_vs_unfused_INT4 | Fused speedup_vs_unfused_INT4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | 117.35 | 38.43 | 40.69 | 3.05x | 2.88x |
| 1 | 128 | 116.17 | 38.04 | 41.88 | 3.05x | 2.77x |
| 1 | 1024 | 112.35 | 40.54 | 43.93 | 2.77x | 2.56x |
| 1 | 4096 | 117.16 | 44.82 | 49.52 | 2.61x | 2.37x |
| 2 | 10 | 134.36 | 39.88 | 42.15 | 3.37x | 3.19x |
| 2 | 128 | 118.35 | 38.83 | 41.59 | 3.05x | 2.85x |
| 2 | 1024 | 117.11 | 40.98 | 44.41 | 2.86x | 2.64x |
| 2 | 4096 | 133.25 | 48.14 | 49.30 | 2.77x | 2.70x |
| 4 | 10 | 133.79 | 38.80 | 41.99 | 3.45x | 3.19x |
| 4 | 128 | 119.12 | 41.31 | 42.55 | 2.88x | 2.80x |
| 4 | 1024 | 123.19 | 43.73 | 44.29 | 2.82x | 2.78x |
| 4 | 4096 | 184.31 | 55.87 | 50.09 | 3.30x | 3.68x |
| 8 | 10 | 116.59 | 39.37 | 43.01 | 2.96x | 2.71x |
| 8 | 128 | 132.14 | 41.10 | 43.24 | 3.22x | 3.06x |
| 8 | 1024 | 132.99 | 47.31 | 45.22 | 2.81x | 2.94x |
| 8 | 4096 | 298.29 | 71.19 | 52.58 | 4.19x | 5.67x |

rocprofv3 代表點：B=1,L=128；B=1,L=4096；B=4,L=1024。

| workload | kernel_calls_per_iter | kernel_time_us_per_iter |
| --- | ---: | ---: |
| fp16_hf_B1_L128 | 4660.500 | 173116.639 |
| fp16_hf_B1_L4096 | 4660.000 | 1703448.907 |
| fp16_hf_B4_L1024 | 4660.000 | 1429737.535 |
| fused_quarot_B1_L128 | 14340.500 | 214751.637 |
| fused_quarot_B1_L4096 | 22276.000 | 1834310.929 |
| fused_quarot_B4_L1024 | 22276.000 | 1564346.307 |
| quarot_unfused_B1_L128 | 27588.500 | 277039.320 |
| quarot_unfused_B1_L4096 | 35524.000 | 1918421.247 |
| quarot_unfused_B4_L1024 | 35524.000 | 1643087.425 |

觀察：

- QuaRot unfused path 的 kernel calls 明顯較高，主要來自 PyTorch separate Hadamard/quant/dequant/copy kernels。
- Fused QuaRot 相對 unfused_INT4 有穩定 `2.37-5.67x` speedup；FP16 參考路徑相對 unfused_INT4 為 `2.61-4.19x`。
- Fused QuaRot 與 FP16 reference 的絕對 latency 在小 shape 接近或略慢，在 B=4,L=4096 與 B=8 長 context 下 fused QuaRot 絕對 latency 低於 FP16 reference。
- 這份結果是 **真實 Llama 權重的 formal token-by-token decode path**，不是 native `transformers.generate()` monkey-patch。

### FP16 HF Full-Model Baseline

Longer-run 設定：

- mode：`fp16_hf`
- batch：`1`
- context length：`10,128,1024`
- warmup：`5`
- iters：`20`
- repeats：`5`
- 結果目錄：`llama31_full_model_results_fp16_B1/`

| batch | context_len | metric | mean (ms) | median (ms) | std | p90 | p95 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | prefill_ms | 41.57 | 41.13 | 0.819 | 42.46 | 42.70 |
| 1 | 10 | decode_ms_per_token | 41.93 | 40.61 | 3.038 | 44.73 | 46.04 |
| 1 | 128 | prefill_ms | 65.57 | 65.54 | 0.132 | 65.72 | 65.74 |
| 1 | 128 | decode_ms_per_token | 40.99 | 40.76 | 0.506 | 41.56 | 41.60 |
| 1 | 1024 | prefill_ms | 205.15 | 205.21 | 0.445 | 205.57 | 205.60 |
| 1 | 1024 | decode_ms_per_token | 43.36 | 42.84 | 0.929 | 44.39 | 44.43 |

Quality self-check：

- 結果目錄：`llama31_full_model_results/`
- `fp16_hf` vs `fp16_hf`
- max/mean error = `0`
- top-1 match = `1.0`
- top-10 overlap = `1.0`
- generation samples 已寫入 `generation_samples.md`

rocprofv3 representative trace：

- workload：`fp16_hf_B1_L128`
- 結果目錄：`llama31_full_model_rocprof/`
- top kernel group 仍以 rocBLAS GEMM (`Cijk_...`) 為主。
- rocprof workload 包含 benchmark warmup/iters 的多次 prefill/decode kernel，因此 `kernel_time_us_per_iter` 主要作為 top-kernel 結構參考，不直接等同 event timing 的單次 latency。

## 舊 Projection-Only Ablation 的定位

早期 `profile_quarot_model_integration.py` 使用 projection 後 tensors 做 ablation，曾量到 `full_fused` 相對 projection-only `unfused_INT4` 約 `100x` event speedup。這個結果只代表 **projection 後 kernel path**，不包含 RMSNorm、QKV/O/FFN linear projection、residual 等 decoder layer 成本。

因此新版報告以 single decoder layer harness 作為主要結論；projection-only ablation 僅保留為 kernel path 上限參考，不作為 full model/end-to-end speedup 宣稱。

## 目前缺少項目

1. **Native Transformers integration**  
   目前已完成 formal token-by-token wrapper，`quarot_unfused` / `fused_quarot` 可透過 `prefill()`、`decode_one()`、wrapper `generate()` 執行。尚未完成的是直接 patch HF `LlamaAttention` / cache class / native `transformers.generate()`，讓一般 HuggingFace model object 不經 wrapper 也能使用 QuaRot paged KV cache + GQA decode。

2. **完整 QuaRot 權重旋轉 / calibration / model conversion**  
   已完成 FP16、QuaRot unfused、Fused QuaRot 的 logits/generation quality 比較；但 FP16 vs QuaRot 差異很大，表示目前還不是正式 QuaRot-converted Llama。下一步需接入真正的 QuaRot 權重旋轉、scale calibration、perplexity 或更大的 prompt set。

3. **更長時間 profiling 與 production 統計**  
   已完成 formal path B=`1,2,4,8`、L=`10,128,1024,4096`、repeats=`3`，並完成 B=1,L=128；B=1,L=4096；B=4,L=1024 三路徑 rocprofv3。正式發表前仍建議增加更長 iters/repeats、多 token decode generation、跨天重跑穩定性與更完整 top-kernel attribution。

4. **硬體 counter 限制**  
   `gfx1201` 上 GL2C/LDS/occupancy counter 仍無法取得有效數值，只能以 kernel time/calls/SQ waves 與理論 traffic 補充。

## Kernel Fusion 可行性評估

目前 kernel fusion 方向可行，而且在 QuaRot original unfused path 上收益明確。主要風險在模型整合邊界，而不是 fused kernel 本身。

- FFN fused kernel 很適合繼續優化，資料流線性、group size 固定、輸出即 packed INT4 + fp16 scale。
- K1 需要持續驗證 paged KV cache layout、page index、last page offset、RoPE position。
- K2 已有可用 INT4 decode kernel，短期重點是確保 K1 產生的 INT4 KV cache layout 完全符合 K2。
- K3 保持獨立 kernel 較安全，後續只需確認下一個 block/linear layer 是否吃 packed INT4 + scale。

## 結論

1. 元件級 profiling 顯示 FFN/K1/K2/K3 fusion 都能顯著降低 unfused_INT4 path 的 kernel calls 與 kernel time；所有 speedup 均以 unfused_INT4 為 `1.00x`。
2. Single decoder layer harness 顯示 fused QuaRot 相對 unfused_INT4 平均約 `6.36x` speedup；FP16 reference 相對 unfused_INT4 平均約 `5.17x`。
3. Llama-3.1 8B formal token-by-token path 顯示 fused QuaRot 相對 unfused_INT4 約 `2.37-5.67x`；FP16 reference 相對 unfused_INT4 約 `2.61-4.19x`。
4. Fused QuaRot 與 FP16 reference 的絕對 latency 是 shape-dependent：小 batch/context 多數接近或略慢，B=4,L=4096 與 B=8 長 context 下 fused QuaRot 絕對 latency 低於 FP16 reference。
5. 目前結果不能外推成 full LLM end-to-end 100x speedup。較精準的說法是：**目前 fused kernels 可有效加速 QuaRot decode path 中 rotation/quant/dequant/INT4 decode 相關 overhead，但整體模型速度仍取決於 GEMM、projection、runtime scheduling、native HF generate 接線與完整 QuaRot model conversion。**

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
