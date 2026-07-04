# Llama-3.1 8B Full Model 接線與驗證計畫

模型：`meta-llama/Llama-3.1-8B`  
狀態：已建立 formal token-by-token full-model evaluation harness；已用 `HF_TOKEN` 成功載入模型，並完成 FP16 / QuaRot unfused / Fused QuaRot 的 latency、quality 與 rocprofv3 代表點。

## 目前已完成

新增目錄：`llama31_quarot/`

- `common.py`：HF model/tokenizer 載入、timing、統計、logits metrics、Llama shape 檢查。
- `model_patch.py`：Llama-3.1 8B 與目前 QuaRot HIP kernels 的相容性檢查。
- `benchmark_full_model.py`：真實 HF FP16 full-model latency benchmark，支援 repeats、p50/p90/p95。
- `validate_quality.py`：真實模型 logits/generation quality 驗證入口。
- `summarize_full_model_results.py`：彙整 latency/quality 成中文報告。
- `profile_full_model_rocprof.sh`：rocprofv3 代表點 profiling wrapper。
- `decode_paths.py`：使用真實 Llama-3.1 8B 權重與 32 層 decoder 的 single-token decode path harness。
- `summarize_decode_paths.py`：彙整 decode-path latency、correctness、rocprof 成中文報告。
- `hf_quarot_model.py`：formal token-by-token wrapper，提供 `prefill()`、`decode_one()`、`generate()`。
- `summarize_formal_full_model.py`：彙整 formal path latency、quality、rocprof 成中文報告。
- `README.md`：執行方式與目前限制。

目前 `fp16_hf` 路徑已是實際 HuggingFace Llama-3.1 8B full-model baseline。  
GQA-aware K1/K2 kernels 已完成；`decode_paths.py` 已驗證真實 Llama 權重的 32 層 decode-step path；`hf_quarot_model.py` 已將該 path 接回 formal token-by-token API，讓 `quarot_unfused`、`fused_quarot` 可透過 wrapper `prefill()`、`decode_one()`、`generate()` 執行。尚未完成的是 native Transformers `LlamaDecoderLayer` / `transformers.generate()` monkey-patch。

## 需要 HF Token

`meta-llama/Llama-3.1-8B` 是 gated repository，需要：

1. HuggingFace 帳號已被授權存取 Meta Llama 3.1。
2. `HF_TOKEN` 有 read 權限。
3. 在 container 內設定：

```bash
export HF_TOKEN=hf_xxx
```

目前腳本已驗證：未設定 token 時會直接停止並提示 `HF_TOKEN is required`，不會開始下載。模型 cache 已放在 workspace 的 `.hf_cache/`，之後 Docker 應掛載：

```bash
-v /user/undergraduate/wfching25/HIP_Fusion/.hf_cache:/root/.cache/huggingface
```

## 關鍵技術差距：GQA

Llama-3.1 8B 的典型 config：

- hidden size = `4096`
- intermediate size = `14336`
- Q heads = `32`
- KV heads = `8`
- head dim = `128`
- layers = `32`

先前 synthetic decoder layer 使用 `Q/K/V heads = 32`，但真實 Llama-3.1 8B 是 GQA：`Q heads=32`、`KV heads=8`。

目前已完成：

1. K1 支援 `[batch, kv_heads=8, head_dim=128]` 的 K/V append。
2. K2 新增 GQA decode：`batch_decode_f16_gqa` / `batch_decode_i4_gqa`。
3. kernel 內正確處理 query head 到 kv head 的 mapping：`kv_head = q_head / 4`。
4. correctness test 已加入 GQA shape：`q_heads=32, kv_heads=8, head_dim=128`。

目前完整 Llama/QuaRot 接線的主要 blocker 已從 kernel/GQA 轉為產品化整合：是否要進一步改成 native HF `LlamaDecoderLayer` patch，讓一般 `transformers.generate()` 直接使用 QuaRot paged KV cache。

## 階段 1：真實 FP16 Baseline

狀態：已完成 B=1 longer-run baseline。

執行：

```bash
cd /workspace/QuaRot
export HF_TOKEN=hf_xxx

python3 -m llama31_quarot.benchmark_full_model \
  --mode fp16_hf \
  --batches 1,2,4 \
  --context-lengths 10,128,1024 \
  --iters 100 \
  --warmup 20 \
  --repeats 5 \
  --out-dir llama31_full_model_results
```

輸出：

- `llama31_full_model_results/model_shape.json`
- `llama31_full_model_results/latency_repeats.csv`
- `llama31_full_model_results/latency_summary.csv`

已完成結果目錄：

- `llama31_full_model_results_smoke/`
- `llama31_full_model_results_fp16_B1/`

目前較穩定的一組結果：

| batch | context_len | metric | mean (ms) | median (ms) | std | p90 | p95 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 10 | prefill_ms | 41.57 | 41.13 | 0.819 | 42.46 | 42.70 |
| 1 | 10 | decode_ms_per_token | 41.93 | 40.61 | 3.038 | 44.73 | 46.04 |
| 1 | 128 | prefill_ms | 65.57 | 65.54 | 0.132 | 65.72 | 65.74 |
| 1 | 128 | decode_ms_per_token | 40.99 | 40.76 | 0.506 | 41.56 | 41.60 |
| 1 | 1024 | prefill_ms | 205.15 | 205.21 | 0.445 | 205.57 | 205.60 |
| 1 | 1024 | decode_ms_per_token | 43.36 | 42.84 | 0.929 | 44.39 | 44.43 |

度量：

- prefill latency
- decode ms/token
- tokens/sec
- repeat mean/median/std/p50/p90/p95

## 階段 2：真實模型輸出品質驗證

狀態：已完成 `fp16_hf` self-check；之後可直接比較 FP16、QuaRot unfused、Fused QuaRot。

執行：

```bash
python3 -m llama31_quarot.validate_quality \
  --candidate-mode fp16_hf \
  --max-lengths 16,128,1024 \
  --max-new-tokens 32 \
  --out-dir llama31_full_model_results
```

輸出：

- `quality_logits.csv`
- `generation_samples.md`

目前 `fp16_hf` vs `fp16_hf` self-check：

- max error = `0`
- mean error = `0`
- top-1 match = `1.0`
- top-10 overlap = `1.0`
- KL divergence 約 `0`，數值上有微小負值是浮點誤差。

品質 metrics：

- logits max error
- logits mean error
- top-1 match
- top-10 overlap
- KL divergence
- greedy generation sample

當 `quarot_unfused` / `fused_quarot` 完成後，固定比較：

1. FP16 HF vs QuaRot unfused
2. QuaRot unfused vs Fused QuaRot
3. FP16 HF vs Fused QuaRot

建議 provisional tolerance：

- fused vs unfused logits top-1 match >= `95%`
- top-10 overlap >= `80%`
- perplexity 不明顯劣於 QuaRot unfused
- generation first mismatch position 需記錄，不用強制全 token 相同

## 階段 3：補 GQA-aware K1/K2

狀態：已完成。

### K1 修改

目前 K1 append kernel 接收 `[batch, heads, 128]`。真實 Llama-3.1 8B 需要：

- key/value input shape = `[batch, kv_heads=8, 128]`
- cache layout 使用 `num_kv_heads`
- API 額外傳入 `num_kv_heads`

測試：

- K1 GQA RoPE off/on packed mismatch rate <= `1%`
- scale max error <= `0.006`
- shapes：batch = `1,2,4,8`，kv_heads = `8`

已跑：

```bash
python3 test_attention_fusion_correctness.py \
  --batch 2 --heads 8 --seq-len 257 --page-size 128 --k3-rows 2,8
```

結果：PASS。

### K2 修改

目前 K2 decode 以 `num_heads` 同時代表 query heads 與 KV heads。GQA 需要：

- query shape = `[batch, q_heads=32, 128]`
- kv cache shape = `[pages, layers, 2, kv_heads=8, page_size, ...]`
- kernel 內 `kv_head = q_head / (q_heads / kv_heads)`

測試：

- FP16 GQA decode reference vs INT4 GQA decode
- L = `10,128,1024,4096`
- batch = `1,2,4,8`
- correctness：attention output max/mean error
- latency：FP16 KV vs INT4 KV

已新增：

- `flashinfer.hip`：`batch_decode_f16_gqa`、`batch_decode_i4_gqa`
- `include_hip/flashinfer/decode.cuh`：`BatchDecodeWithPagedKVGQAKernel`
- `bench_flashinfer_gqa.py`

完整結果：

- `flashinfer_gqa_results/gqa_decode.csv`
- `flashinfer_gqa_results/summary.md`
- `rocprof_flashinfer_gqa/summary.md`

全部 GQA f16/i4 decode 相對 expanded-reference 的 max/mean error = `0`。

## 階段 4：完整 Llama/QuaRot Model Class 接線

狀態：已完成 formal token-by-token wrapper；native HF `transformers.generate()` monkey-patch 尚未完成。

已完成內容：

- 使用 `meta-llama/Llama-3.1-8B` 真實權重與 tokenizer。
- Prefill 由 HF model 產生每層 prompt KV cache。
- `QuaRotLlamaForCausalLM` 提供 `prefill()`、`decode_one()`、`generate()`。
- Decode step 跑完整 32 層 decoder：RMSNorm、QKV projection、Llama RoPE、KV append、attention decode、O projection、residual、FFN gate/up/down projection、final norm/lm head。
- 三條路徑：
  - `fp16_hf`：HF prefill + FP16 paged KV + GQA-aware FP16 decode。
  - `quarot_unfused`：PyTorch Hadamard/INT4 pack/dequant + GQA-aware FP16 decode。
  - `fused_quarot`：HIP K1/K2/K3/FFN fused kernels。
- 早期 `decode_paths.py` 驗證中，manual FP16 decode 已與 HF `fp16_hf` decode logits 對齊，top1/top10 match = `1.0`。
- 產物：
  - `llama31_decode_path_results/`
  - `llama31_decode_path_results_L4096/`
  - `llama31_decode_path_report_zh.md`
  - `llama31_formal_latency_fp16/`
  - `llama31_formal_latency_quarot_unfused/`
  - `llama31_formal_latency_fused_quarot/`
  - `llama31_formal_quality_fp16_vs_unfused/`
  - `llama31_formal_quality_fp16_vs_fused/`
  - `llama31_formal_quality_unfused_vs_fused/`
  - `llama31_formal_full_model_report_zh.md`

接線策略是最小侵入式 wrapper，而不是直接改 Transformers 原始 class。

保留 HF 原始模組：

- `LlamaRMSNorm`
- `q_proj/k_proj/v_proj/o_proj`
- `gate_proj/up_proj/down_proj`
- residual graph
- tokenizer/generation loop

替換部分：

- K1：projected K/V append to INT4 paged cache
- K2：GQA-aware INT4 decode
- K3：attention output Hadamard + INT4 quant
- FFN：`SiLU(gate) * up -> Hadamard -> INT4 pack`

新增 modes：

- `fp16_hf`
- `quarot_unfused`
- `fused_quarot`

里程碑：

1. patch 單一 decoder layer，固定一個 prompt、一個 decode token。`已由 manual layer loop 覆蓋`
2. patch 所有 decoder layers，但只跑 greedy decode 1 token。`已由 decode_paths.py 覆蓋`
3. 接回 `generate()` 或自寫 token-by-token decode loop。`已由 QuaRotLlamaForCausalLM formal API 完成`
4. 跑完整 latency/quality/profiling grid。`已完成 B=1/2/4/8, L=10/128/1024/4096 與指定 rocprof 代表點`

## 階段 5：更長時間 Profiling 與多次重跑

狀態：已完成 formal path 多次重跑與指定 rocprofv3；更長 iters/repeats 與資料集品質評估保留為後續工作。

已完成 event benchmark：

- warmup = `1`
- iters = `3`
- repeats = `3`
- batch = `1,2,4,8`
- context length = `10,128,1024,4096`
- 報告 mean、median、std、p50、p90、p95、min、max

代表結果：

| batch | context_len | FP16 HF (ms/token) | QuaRot unfused (ms/token) | Fused QuaRot (ms/token) | fused vs unfused | fused vs FP16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 38.04 | 116.2 | 41.88 | 2.77x | 0.91x |
| 1 | 4096 | 44.82 | 117.2 | 49.52 | 2.37x | 0.91x |
| 4 | 1024 | 43.73 | 123.2 | 44.29 | 2.78x | 0.99x |
| 8 | 4096 | 71.19 | 298.3 | 52.58 | 5.67x | 1.35x |

已完成 rocprofv3：

- 代表點：B=1,L=128；B=1,L=4096；B=4,L=1024
- 路徑：`fp16_hf`、`quarot_unfused`、`fused_quarot`
- 輸出：`llama31_formal_rocprof/summary.md`

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

原規劃的 longer-run 目標仍建議後續補齊：

- warmup = `20`
- iters = `100`
- repeats = `5`
- 報告 mean、median、std、p50、p90、p95

Shapes：

- batch = `1,2,4,8`
- context length = `10,128,1024,4096`
- decode tokens = `1,16,64,128`

rocprofv3 代表點：

- B=1, L=128
- B=1, L=4096
- B=4, L=1024

三條路徑都跑：

- FP16 HF baseline
- QuaRot original unfused
- Fused QuaRot

若 `gfx1201` GL2C/LDS/occupancy counter 仍為 0：

- 報告明確註記工具限制
- 使用 kernel time / kernel calls / top kernels
- 補理論最小 memory traffic

## 後續工作

已完成把 `decode_paths.py` 中驗證過的 32 層 decode path 接回 formal token-by-token API，並已重跑：

1. full-model logits/generation quality：FP16 vs QuaRot unfused vs Fused QuaRot。
2. full-model latency：batch `1,2,4,8`，context `10,128,1024,4096`。
3. rocprofv3：B=1,L=128；B=1,L=4096；B=4,L=1024。

後續若要再往正式 model integration 推進，建議優先做：

1. Native HF `LlamaDecoderLayer` / `Cache` monkey-patch，讓 `transformers.generate()` 不需 wrapper 也可使用 QuaRot path。
2. 正式 QuaRot 權重旋轉、scale calibration、model conversion。
3. 更大 prompt set / perplexity / first-mismatch generation analysis。
4. 更長 iters/repeats 與跨次重跑 profiling，補足 production-grade 統計。
