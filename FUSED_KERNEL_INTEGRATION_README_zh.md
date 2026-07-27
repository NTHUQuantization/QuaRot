# Fused QuaRot Kernels 整合指南

本文件提供給負責 Llama/QuaRot model integration 的組員，說明如何把目前 HIP fused kernels 接入 Llama-3.1 8B token-by-token decode。

## 先讀結論

整合者有兩種使用方式：

1. **先驗證功能：直接使用現有 formal wrapper**
   在本 repo 內使用 `llama31_quarot/hf_quarot_model.py` 的 `QuaRotLlamaForCausalLM`。這是目前 benchmark 與 profiling 採用的相同路徑。
2. **接入自己的 model class：保留 kernel source tree，依本文件在 decoder layer 的四個位置呼叫 K1/K2/K3/FFN API**
   不建議只複製編譯後 `.so`；`.so` 綁定 Python、PyTorch、ROCm ABI 與 GPU architecture，應複製 source 後在目標環境重編。

目前 production/default backend 是 `current`。`hadacore256` 是實驗 backend，完整 decode 沒有穩定勝過 current，不應作為整合預設值。

## 必須帶走哪些檔案

如果整合工作仍在本 repo 內，**不需要搬檔案**，直接從 `/workspace/QuaRot` build/import 即可。

如果要 vendor 到另一個 project，建議在目標 repo 建立：

```text
your_project/
  third_party/
    fused_quarot/
      setup_flashinfer.py
      flashinfer.hip
      hadamard_wmma.cuh
      include_hip/
        flashinfer/                 # 整個目錄都要保留
      flashinfer_test/
        __init__.py                 # 可為空檔案
      attention_fusion/
        __init__.py
        setup.py
        attention_fusion.hip
        quarot_attention_fusion.py
      ffn_fusion/
        setup.py
        ffn_fusion.hip
```

來源對照：

| 用途 | 必要來源 |
| --- | --- |
| K1 / K3 | `attention_fusion/attention_fusion.hip`、`attention_fusion/quarot_attention_fusion.py`、`attention_fusion/__init__.py`、`attention_fusion/setup.py` |
| K2 GQA INT4 decode | `flashinfer.hip`、`setup_flashinfer.py`、完整 `include_hip/flashinfer/` |
| FFN | `ffn_fusion/ffn_fusion.hip`、`ffn_fusion/setup.py` |
| Hadamard device helper | `hadamard_wmma.cuh`；K1/K3與FFN source以 `../hadamard_wmma.cuh` 引用它 |
| 正式接線參考 | `llama31_quarot/hf_quarot_model.py` |
| Active paged-cache metadata | `make_active_metadata()`，位於 `llama31_quarot/hf_quarot_model.py` |
| Correctness tests | `attention_fusion/test_attention_fusion_correctness.py`、`ffn_fusion/bench_ffn_fusion.py`、`test_quarot_model_integration.py` |

不要把以下檔案當成 runtime dependency 搬走：

- `build/`
- `*.o`
- 現有 `*.so`
- profiling CSV、rocprof traces
- `hadacore_variant_results/`

## Build

目前 target 是 `gfx1201`、ROCm 7.2、PyTorch 2.9。以下命令在 ROCm/PyTorch container 內執行。

```bash
cd /workspace/QuaRot

# K2: flashinfer_test._HIP
python3 setup_flashinfer.py build_ext --inplace

# K1 / K3: attention_fusion.attention_fusion_hip
cd /workspace/QuaRot/attention_fusion
ATTN_FUSION_HIP_ARCHS=gfx1201 python3 setup.py build_ext --inplace

# FFN: ffn_fusion_hip
cd /workspace/QuaRot/ffn_fusion
FFN_FUSION_HIP_ARCHS=gfx1201 python3 setup.py build_ext --inplace
```

若換 GPU，先以 `rocminfo` 查 architecture，再改 `*_HIP_ARCHS`；`setup_flashinfer.py` 目前把 `gfx1201` 寫死，非 gfx1201 時也要同步修改其 `--offload-arch`。

Import smoke test：

```bash
cd /workspace/QuaRot
PYTHONPATH=/workspace/QuaRot:/workspace/QuaRot/ffn_fusion \
python3 -c "import flashinfer_test._HIP; import attention_fusion; import ffn_fusion_hip; print('imports: PASS')"
```

目標 project runtime 也必須讓 fused QuaRot root 與 `ffn_fusion/` 在 `PYTHONPATH`：

```bash
export PYTHONPATH=/path/to/your_project/third_party/fused_quarot:/path/to/your_project/third_party/fused_quarot/ffn_fusion:$PYTHONPATH
```

## 支援範圍與 Tensor Contract

目前正式測試 shape 是 Llama-3.1 8B：

| 項目 | 值 |
| --- | ---: |
| hidden size | 4096 |
| FFN intermediate | 14336；prototype亦測過11008 |
| query heads | 32 |
| KV heads | 8 |
| head dim | 128 |
| page size | 128 |
| K3/FFN group size | 256 |

所有 input tensor 都必須：

- 在同一張 HIP device；PyTorch API 仍使用 device string `"cuda"`。
- contiguous。
- 除了 packed payload/index外為 `torch.float16`。
- metadata index為 `torch.int32`。

### Paged KV cache layout

```text
kv_data:
  shape [total_pages, num_layers, 2, kv_heads, page_size, 64]
  dtype uint8
  每 byte 存兩個 signed INT4 values

kv_param:
  shape [total_pages, num_layers, 2, kv_heads, page_size, 2]
  dtype float16
  最後一維是 kernel 使用的 scale metadata

metadata.indptr / indices / last_page_offset:
  dtype int32
```

`2` 這一維依序代表 K、V。Llama-3.1 8B cache 必須依 `kv_heads=8` 配置，不是 `q_heads=32`。

## 應插入 Decoder Layer 的位置

```text
RMSNorm
  -> Q/K/V projections + RoPE
  -> K1: K/V FP16 -> Hadamard -> INT4 quant/pack -> paged cache append
  -> K2: Q FP16 + paged INT4 KV cache -> attention output FP16
  -> K3: attention output FP16 -> grouped Hadamard -> INT4 packed + scales
  -> [目前 HF path: dequant K3 -> FP16 O projection -> residual]
  -> post-attention RMSNorm
  -> gate/up projections FP16
  -> FFN: SiLU(gate) * up -> grouped Hadamard -> INT4 packed + scales
  -> [目前 HF path: dequant FFN -> FP16 down projection -> residual]
```

K1/K2/K3/FFN 都是 **decode-only integration**。Prefill 仍由 HF model 建立 cache，再轉成 local paged INT4 layout；不可把單 token K1 append 當成完整 prefill kernel。

## 最快可用方式：Formal Wrapper

若整合者只需要先跑通相同路徑：

```python
from llama31_quarot.hf_quarot_model import wrap_model

# model 是 dtype=torch.float16、已放上 HIP device 的 HF LlamaForCausalLM。
quarot = wrap_model(model, mode="fused_quarot", fusion_backend="current")

prefill_logits, cache = quarot.prefill(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=32,
)

# 每次只接受 [batch, 1]
decode_logits, cache = quarot.decode_one(next_token_ids, cache)
```

這個 wrapper：

- prefill 使用 HF model。
- 將 prompt KV cache 轉成 paged INT4 cache。
- 32 層逐層呼叫 GQA-aware K1/K2/K3/FFN。
- 可用 `generate()` 做 greedy token-by-token generation。

它不是 native `transformers.generate()` monkey patch；若服務程式一定要使用 HF `DynamicCache` 或原生 `generate()`，請採下一節的 decoder-layer 接法。

## 接入自己的 LlamaDecoderLayer

### 1. Import

```python
import torch
import flashinfer_test._HIP as flashinfer_hip
import ffn_fusion_hip

from attention_fusion import (
    PagedKVMetadata,
    allocate_quantized_kv_cache,
    append_quantized_kv_decode,
    make_uniform_paged_kv_metadata,
    quantize_attention_output,
)
```

### 2. Cache allocation

在 request/cache object 建立時配置，不要每 layer、每 token 重配：

```python
allocation = make_uniform_paged_kv_metadata(
    batch_size=batch,
    seq_len=max_seq_len,
    page_size=128,
    device=device,
)

kv_data, kv_param = allocate_quantized_kv_cache(
    allocation,
    num_layers=32,
    num_heads=8,       # Llama-3.1 8B KV heads
    device=device,
)
```

`make_uniform_paged_kv_metadata(..., max_seq_len)` 描述的是容量配置。每次 decode 必須傳入只暴露目前有效 prefix 的 active metadata；可直接複用：

```python
from llama31_quarot.hf_quarot_model import make_active_metadata

active = make_active_metadata(allocation, current_seq_len, device)
```

若把 cache 接到 scheduler，應讓 scheduler 維護 `indptr/indices/last_page_offset`，而不是每 token 用 Python 重建。

### 3. K1：append quantized K/V

插在 Q/K/V projection與RoPE之後：

```python
# k, v: [B, 8, 128], FP16, contiguous
append_quantized_kv_decode(
    k.contiguous(),
    v.contiguous(),
    active,
    kv_data,
    kv_param,
    num_layers=32,
    layer_idx=layer_idx,
    apply_rope_to_k=False,
)
```

目前 formal path 已由 HuggingFace `apply_rotary_pos_emb()` 對 K 做 RoPE，所以必須使用 `apply_rope_to_k=False`。只有在 K 尚未做 RoPE 時才可打開 K1 RoPE，否則會套用兩次。

### 4. K2：GQA-aware INT4 attention decode

Llama-3.1 8B 必須呼叫 `batch_decode_i4_gqa`：

```python
# q: [B, 32, 128], FP16
attn_out = torch.empty_like(q)

flashinfer_hip.batch_decode_i4_gqa(
    attn_out,
    q.contiguous(),
    kv_data,
    kv_param,
    active.indptr,
    active.indices,
    active.last_page_offset,
    32,                 # num_layers
    layer_idx,
    32,                 # num_q_heads
    8,                  # num_kv_heads
    128,                # page_size
    batch,
)
```

不要使用舊 `QuaRotFusedDecodeLayer.decode_attention()` 作為 Llama-3.1 正式接法；該 helper 使用非 GQA `batch_decode_i4`，只適合 query heads與KV heads相同的舊 synthetic path。

### 5. K3：attention output quantization

```python
# attn_out 原為 [B, 32, 128]
k3_packed, k3_scales = quantize_attention_output(
    attn_out.reshape(batch, 4096).contiguous(),
    backend="current",
)

# output:
# k3_packed [B, 2048] uint8
# k3_scales [B, 16] float16
```

目前原始 HF `o_proj` 只接受 FP16，因此現階段需先用既有 `dequant_grouped()` reference 回 FP16；正式 W4A4 integration 應改成讓 packed K3 output直接進 INT4 rotated O projection。詳見 `int4_coverage_audit_zh.md`。

### 6. FFN fused kernel

插在 gate/up projections之後、down projection之前：

```python
# gate/up: [B, 14336] FP16 contiguous
ffn_packed, ffn_scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(
    gate.reshape(batch, -1).contiguous(),
    up.reshape(batch, -1).contiguous(),
    256,
)

# output:
# ffn_packed [B, 7168] uint8 for hidden=14336
# ffn_scales [B, 56] float16
```

目前 HF `down_proj` 同樣需先 dequant回 FP16。完整整合的下一步是 INT4 activation × INT4 rotated down weight GEMM，而不是在 kernel後永久保留 `ffn_dequant`。

## 不要犯的接線錯誤

1. **Llama-3.1 使用非 GQA K2 API**：Q heads=32、KV heads=8，必須使用 `batch_decode_i4_gqa`。
2. **Cache以32 heads配置**：會造成 layout與容量錯誤；cache head dimension必須是8。
3. **RoPE做兩次**：HF已做 RoPE時，K1的 `apply_rope_to_k` 必須是 `False`。
4. **固定使用 max-length metadata做 decode**：K2會讀到尚未寫入的 pages/slots；每步要使用 active prefix metadata。
5. **每 token重新 allocate cache**：會抹掉歷史 KV 並讓 latency失真。
6. **把 packed `uint8` 當 unsigned INT8**：每個 byte內是兩個 offset-binary signed INT4 nibbles，不能直接 `.to(float16)`。
7. **只複製 `.so`**：PyTorch/ROCm/Python ABI或GPU arch不同時可能 import失敗或執行錯誤。
8. **把 K3/FFN packed output直接餵原始 HF linear**：HF linear不認得 packed layout，且 weights尚未做對應 QuaRot rotation。
9. **宣稱已是完整 QuaRot W4A4**：目前只有 KV cache長期維持INT4；projection weights仍是FP16。
10. **預設使用 hadacore backend**：目前 full decode沒有穩定效能優勢，先使用 `current`。

## Correctness 驗收

Build完成後至少執行：

```bash
cd /workspace/QuaRot/attention_fusion
python3 test_attention_fusion_correctness.py \
  --batch 2 --heads 8 --seq-len 257 --page-size 128 --k3-rows 2,8

cd /workspace/QuaRot/ffn_fusion
python3 bench_ffn_fusion.py \
  --rows 2 --cols 14336 --group-size 256 --backend current \
  --warmup 10 --iters 100 --ref-iters 10

cd /workspace/QuaRot
PYTHONPATH=/workspace/QuaRot:/workspace/QuaRot/ffn_fusion \
python3 test_quarot_model_integration.py \
  --batch 2 --ffn-hidden 14336
```

正式 model integration 還必須比較：

- 每層 hidden state max/mean/relative error。
- final logits max/mean/relative error。
- top-1 agreement、top-10 overlap。
- prompt prefill後第一個decode token，以及連續至少32 token generation。
- B=1/2/4/8，L=10/128/1024/4096。
- cache `seq_len`、active page count、last page offset是否逐token正確增加。

## 效能驗收

- Timing範圍只能包含decode，不含model load、prefill、cache conversion與warmup。
- 使用HIP event量完整token latency，不只量單一kernel。
- rocprof region應從warmup完成後才Resume，decode完成同步後立即Pause。
- 所有speedup以相同權重/shape的 `unfused_INT4` 為baseline。
- `current` backend應先作正式default；hadacore只作獨立ablation。

## 目前限制

- Kernels固定針對 `head_dim=128`、hidden=4096、group=256設計。
- K1/K2是decode path；沒有完整fused prefill。
- Formal wrapper尚未替換原生HF cache class或直接patch `transformers.generate()`。
- Q與K/V目前是mixed FP16×INT4 K2，不是純Q4×KV4。
- K3/FFN packed output仍在HF projection前dequant。
- Model weights尚未完成正式QuaRot rotation、INT4 conversion與calibration。
- 因此整合者應先重現現有formal path，再進行native cache與W4A4 GEMM接線，不能跳過rotation correctness。

## 主要參考檔

- `llama31_quarot/hf_quarot_model.py`：正式32層token decode接線。
- `attention_fusion/quarot_attention_fusion.py`：K1/K3穩定Python API與cache allocation。
- `flashinfer.hip`：K2 GQA INT4 binding。
- `ffn_fusion/ffn_fusion.hip`：FFN fused kernel binding。
- `int4_coverage_audit_zh.md`：仍為FP16的區塊與後續W4A4路線。
- `decode_bottleneck_profiling_report_no_hadacore_zh.md`：unfused INT4對fused current的完整效能分析。
