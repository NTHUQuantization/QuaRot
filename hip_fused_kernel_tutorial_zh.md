# HIP Fused Kernel 實作教學：QuaRot FFN / K1 / K2 / K3

本文是給 trace code 用的實作導讀，重點放在本 repo 的 HIP fused kernels 如何把 QuaRot decode path 中的 Hadamard rotation、dynamic INT4 quantization、packing、paged KV cache append 和 INT4 attention decode 串起來。

對應主要檔案：

- `quarot_model_integration.py`：Python 層把 K1/K2/K3/FFN 串成 decode step。
- `attention_fusion/attention_fusion.hip`：K1 和 K3 fused kernels。
- `flashinfer.hip`：K2 FlashInfer INT4 paged KV decode wrapper。
- `ffn_fusion/ffn_fusion.hip`：FFN fused kernel。
- `attention_fusion/quarot_attention_fusion.py`：K1/K3 Python-facing helper。
- `hadamard_wmma.cuh`：Hadacore / WMMA Hadamard helper。

## 1. Fused QuaRot Decode Path 總覽

整體 fused path 在 `QuaRotFusedDecodeLayer` 裡描述，見 `quarot_model_integration.py:33`。Python docstring 已把四個 fused block 的責任列出：

```python
K1: append K/V with Hadamard + dynamic INT4 quantization
K2: FlashInfer INT4 KV decode
K3: attention output Hadamard + dynamic INT4 quantization
FFN: SiLU(gate) * up + Hadamard + dynamic INT4 quantization
```

真正的 decode step 順序在 `quarot_model_integration.py:148`：

```python
self.append_kv(key, value, apply_rope_to_k=apply_rope_to_k)
attention = self.decode_attention(query)
attention_packed, attention_scales = self.quantize_attention(attention)
ffn_packed, ffn_scales = self.quantize_ffn(ffn_gate, ffn_up)
```

這四行分別對應：

| 步驟 | 函式 | 實作檔案 | 主要功能 |
| --- | --- | --- | --- |
| K1 | `append_quantized_kv_decode` | `attention_fusion/attention_fusion.hip` | K/V append 到 paged KV cache，內含 optional RoPE、Hadamard、INT4 quant、pack |
| K2 | `flashinfer_hip.batch_decode_i4` | `flashinfer.hip` + FlashInfer headers | 從 INT4 paged KV cache 做 decode attention |
| K3 | `quantize_attention_output` | `attention_fusion/attention_fusion.hip` | attention output 做 grouped Hadamard + INT4 quant |
| FFN | `fused_ffn_silu_hadamard_quant` | `ffn_fusion/ffn_fusion.hip` | `SiLU(gate) * up` 後做 Hadamard + INT4 quant |

核心概念是：unfused path 會把 Hadamard、amax、scale、round、clamp、pack 拆成很多 PyTorch operation 或小 kernel；fused path 則讓一個 HIP kernel 在 shared memory 裡完成整串流程，最後只寫出 packed INT4 和 scale。

## 2. 共用資料格式

### 2.1 Signed INT4 Pack

K1、K3、FFN 都使用同一種 signed INT4 packing。helper 在：

- `ffn_fusion/ffn_fusion.hip:21`
- `attention_fusion/attention_fusion.hip:21`

程式片段：

```cpp
__device__ __forceinline__ uint8_t pack_s4_pair(int lo, int hi) {
  uint8_t ulo = static_cast<uint8_t>(lo + 8) & 0x0f;
  uint8_t uhi = static_cast<uint8_t>(hi + 8) & 0x0f;
  return static_cast<uint8_t>(ulo | (uhi << 4));
}
```

語意：

- 量化後的 logical value 是 signed INT4：`[-8, 7]`。
- 儲存時加上 offset `+8`，變成 unsigned nibble `[0, 15]`。
- `lo` 放在 byte 的低 4 bits。
- `hi` 放在 byte 的高 4 bits。
- 所以兩個元素共用一個 `uint8_t`。

這也是為什麼很多 kernel 都有：

```cpp
if ((tid & 1) == 0) {
  ...
  packed[...] = pack_s4_pair(q0, q1);
}
```

只有偶數 thread 寫入，因為一個偶數 thread 負責 `(tid, tid + 1)` 這一組 pair。

### 2.2 Dynamic INT4 Scale

量化 scale 的基本公式是：

```cpp
scale = max(abs(x)) / 7
```

並且會 clamp 下限：

```cpp
float scale = fmaxf(reduce[0] / 7.0f, 1.0e-8f);
```

為什麼除以 `7` 而不是 `8`：

- signed INT4 positive endpoint 是 `+7`。
- negative endpoint 是 `-8`。
- 用 `amax / 7` 能讓最大正值對齊到 `+7`。
- 之後 round 再 clamp 到 `[-8, 7]`。

量化片段通常長這樣：

```cpp
int q0 = static_cast<int>(nearbyintf(values[tid] / scale));
q0 = max(-8, min(7, q0));
```

### 2.3 Hadamard Transform Pattern

目前 current backend 的 Hadamard 是 butterfly 寫法。共用 helper 在 `attention_fusion/attention_fusion.hip:29`：

```cpp
for (int stride = 1; stride < n; stride <<= 1) {
  int tid = threadIdx.x;
  int pair_base = ((tid / stride) * (stride << 1)) + (tid & (stride - 1));
  if (tid < (n >> 1)) {
    float a = data[pair_base];
    float b = data[pair_base + stride];
    data[pair_base] = a + b;
    data[pair_base + stride] = a - b;
  }
  __syncthreads();
}
```

重點：

- 每一輪 `stride` 都把距離為 `stride` 的兩個元素做 `a+b` / `a-b`。
- 只有前半 thread active，因為一個 active thread 會寫兩個位置。
- 每一輪後要 `__syncthreads()`，否則下一輪會讀到尚未完成的 shared memory。
- 最後通常乘上 `rsqrtf(n)`，做 normalized / orthonormal Hadamard。

## 3. FFN Fused Kernel

### 3.1 Python 入口

在 `quarot_model_integration.py:123`：

```python
def quantize_ffn(self, gate: torch.Tensor, up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.ffn_backend == "current":
        packed, scales = ffn_fusion_hip.fused_ffn_silu_hadamard_quant(
            gate.contiguous(), up.contiguous(), self.ffn_group_size
        )
```

這會呼叫 C++/HIP extension 裡的 `fused_ffn_silu_hadamard_quant`，binding 在 `ffn_fusion/ffn_fusion.hip:307`。

### 3.2 Kernel 入口與 Launch

current kernel 在 `ffn_fusion/ffn_fusion.hip:29`：

```cpp
template <int GroupSize>
__global__ void fused_ffn_silu_hadamard_quant_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    uint8_t* __restrict__ packed,
    half* __restrict__ scales,
    int rows,
    int cols)
```

launch 設定在 `ffn_fusion/ffn_fusion.hip:184`：

```cpp
dim3 grid(rows, cols / GroupSize);
dim3 block(GroupSize);
size_t smem = static_cast<size_t>(GroupSize) * 2 * sizeof(float);
```

對應關係：

- `blockIdx.x`：flattened row，例如 batch/token row。
- `blockIdx.y`：hidden dimension 上的 group。
- `threadIdx.x`：group 內第幾個元素。
- `blockDim.x = GroupSize`。
- shared memory 分成兩段：`values` 和 `reduce`。

### 3.3 FFN Kernel 的完整資料流

#### Step 1：讀 gate/up 並計算 activation

見 `ffn_fusion/ffn_fusion.hip:50`：

```cpp
float g = __half2float(gate[base + tid]);
float u = __half2float(up[base + tid]);
values[tid] = silu(g) * u;
```

這對應 SwiGLU / gated FFN 的中間 activation：

```text
ffn_intermediate = SiLU(gate) * up
```

unfused path 通常會先產生一個完整 FP16/FP32 intermediate tensor；fused kernel 則直接放進 shared memory。

#### Step 2：在 shared memory 做 Hadamard

見 `ffn_fusion/ffn_fusion.hip:56`：

```cpp
for (int stride = 1; stride < GroupSize; stride <<= 1) {
  int pair_base = ((tid / stride) * (stride << 1)) + (tid & (stride - 1));
  if (tid < (GroupSize >> 1)) {
    float a = values[pair_base];
    float b = values[pair_base + stride];
    values[pair_base] = a + b;
    values[pair_base + stride] = a - b;
  }
  __syncthreads();
}
```

這裡是 in-place fast Walsh-Hadamard transform。因為 `values` 在 shared memory，所以不需要把每一輪 Hadamard 的中間結果寫回 global memory。

#### Step 3：Normalize

見 `ffn_fusion/ffn_fusion.hip:69`：

```cpp
const float norm = rsqrtf(static_cast<float>(GroupSize));
float rotated = values[tid] * norm;
values[tid] = rotated;
```

QuaRot 的 rotation 希望保持能量尺度穩定，所以 Hadamard 後乘 `1 / sqrt(GroupSize)`。

#### Step 4：Reduce amax

見 `ffn_fusion/ffn_fusion.hip:74` 到 `ffn_fusion/ffn_fusion.hip:82`：

```cpp
reduce[tid] = fabsf(rotated);
__syncthreads();

for (int stride = GroupSize >> 1; stride > 0; stride >>= 1) {
  if (tid < stride) {
    reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
  }
  __syncthreads();
}
```

這是 block 內 reduction，最後 `reduce[0]` 是這個 group 的 absolute max。

#### Step 5：Scale + Quant + Pack

見 `ffn_fusion/ffn_fusion.hip:84`：

```cpp
float scale = fmaxf(reduce[0] / 7.0f, 1.0e-8f);
if (tid == 0) {
  scales[row * gridDim.y + group] = __float2half(scale);
}
```

見 `ffn_fusion/ffn_fusion.hip:91`：

```cpp
if ((tid & 1) == 0) {
  int q0 = static_cast<int>(nearbyintf(values[tid] / scale));
  int q1 = static_cast<int>(nearbyintf(values[tid + 1] / scale));
  q0 = max(-8, min(7, q0));
  q1 = max(-8, min(7, q1));
  packed[(row * cols + group * GroupSize + tid) >> 1] = pack_s4_pair(q0, q1);
}
```

輸出 shape：

- `packed`：最後一維從 `cols` 變成 `cols / 2`。
- `scales`：最後一維從 `cols` 變成 `cols / group_size`。

### 3.4 FFN Hadacore256 版本

Hadacore 版本在 `ffn_fusion/ffn_fusion.hip:102`：

```cpp
__global__ void fused_ffn_silu_hadamard_quant_hadacore256_kernel(...)
```

差異：

- 固定 group size = 256。
- block 只有 32 threads，一個 wave。
- 每個 lane 處理多個元素：

```cpp
for (int i = lane; i < 256; i += 32) {
  ...
}
```

核心 Hadamard 呼叫在 `ffn_fusion/ffn_fusion.hip:128`：

```cpp
quarot_hadamard_wmma::h256_f16(in_bits, temp_bits, out_bits, lane);
```

這條路徑的目的不是改變語意，而是用 WMMA helper 加速 256-point Hadamard。

## 4. K1：Append KV + RoPE + Hadamard + INT4 Quant

### 4.1 Python 入口

在 `quarot_model_integration.py:85`：

```python
def append_kv(self, key: torch.Tensor, value: torch.Tensor, *, apply_rope_to_k: bool = False) -> None:
    self.kv_data, self.kv_param = append_quantized_kv_decode(...)
```

`append_quantized_kv_decode` 在 `attention_fusion/quarot_attention_fusion.py:58`，最後會呼叫：

```python
attention_fusion_hip.append_kv_had_quant_inplace(...)
```

### 4.2 Kernel 入口與 Launch

K1 kernel 在 `attention_fusion/attention_fusion.hip:70`：

```cpp
__global__ void append_kv_had_quant_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    uint8_t* __restrict__ kv_data,
    half2* __restrict__ kv_param,
    ...
)
```

launch 設定在 `attention_fusion/attention_fusion.hip:449`：

```cpp
dim3 grid(static_cast<unsigned int>(batch_size), static_cast<unsigned int>(num_heads));
dim3 block(kHeadDim);
```

對應關係：

- `blockIdx.x = batch`
- `blockIdx.y = head`
- `threadIdx.x = head_dim element`
- `kHeadDim = 128`

所以一個 block 負責一個 batch 的一個 head 的 current token K/V。

### 4.3 Paged KV Cache 位置計算

K1 最容易 trace 錯的是 paged KV cache offset。關鍵在 `attention_fusion/attention_fusion.hip:98`：

```cpp
int seq_len = (kv_indptr[batch + 1] - kv_indptr[batch] - 1) * page_size + last_page_offset[batch];
int page_idx = kv_indices[kv_indptr[batch] + (seq_len - 1) / page_size];
int entry_idx = (seq_len - 1) % page_size;
```

解讀：

- `kv_indptr` 和 `kv_indices` 描述每個 batch 對應哪些 physical pages。
- `last_page_offset[batch]` 是最後一頁目前用了多少 token slot。
- `seq_len - 1` 是 current appended token 的 position。
- `page_idx` 是 physical page id。
- `entry_idx` 是 page 內 offset。

K/V packed data layout 在 allocation 時決定，見 `attention_fusion/quarot_attention_fusion.py:45`：

```python
kv_data = torch.empty(
    (metadata.total_pages, num_layers, 2, num_heads, metadata.page_size, HEAD_DIM // 2),
    ...
)
kv_param = torch.empty(
    (metadata.total_pages, num_layers, 2, num_heads, metadata.page_size, 2),
    ...
)
```

layout 維度語意：

```text
kv_data[page, layer, k_or_v, head, token_slot, packed_head_dim]
kv_param[page, layer, k_or_v, head, token_slot, 2]
```

其中：

- `k_or_v = 0` 表示 K。
- `k_or_v = 1` 表示 V。
- `packed_head_dim = 128 / 2 = 64 bytes`。
- `kv_param[..., 0] = scale`。
- `kv_param[..., 1] = scale * 8`。

### 4.4 Optional RoPE

見 `attention_fusion/attention_fusion.hip:105`：

```cpp
float kval = __half2float(key[input_offset]);
if (apply_rope_to_k) {
  int pair_idx = tid >> 1;
  int pair_base = (batch * num_heads + head) * kHeadDim + (pair_idx << 1);
  float x0 = __half2float(key[pair_base]);
  float x1 = __half2float(key[pair_base + 1]);
  rope_pair(x0, x1, pair_idx, seq_len - 1, rope_inv_theta);
  kval = (tid & 1) ? x1 : x0;
}
```

RoPE 是以 even/odd pair 做旋轉：

```cpp
y0 = x0 * c - x1 * s;
y1 = x0 * s + x1 * c;
```

注意：K1 的 RoPE 是 optional，因為某些 integration path 可能已經在進 kernel 前做過 RoPE。

### 4.5 K 的 Hadamard + Quant + 寫入 cache

見 `attention_fusion/attention_fusion.hip:120`：

```cpp
hadamard_inplace(k_smem, kHeadDim);
float k_rot = k_smem[tid] * rsqrtf(static_cast<float>(kHeadDim));
k_smem[tid] = k_rot;
float k_scale = reduce_amax(k_rot, reduce);
```

寫 packed K 的 offset 在 `attention_fusion/attention_fusion.hip:127`：

```cpp
if ((tid & 1) == 0) {
  int kq1 = max(-8, min(7, static_cast<int>(nearbyintf(k_smem[tid + 1] / k_scale))));
  size_t k_elem = (((page_idx * num_layers + layer_idx) * 2 * num_heads + head) * page_size + entry_idx) *
                  (kHeadDim / 2) + (tid >> 1);
  kv_data[k_elem] = pack_s4_pair(kq, kq1);
}
```

寫 scale metadata 在 `attention_fusion/attention_fusion.hip:133`：

```cpp
if (tid == 0) {
  size_t k_param = ((page_idx * num_layers + layer_idx) * 2 * num_heads + head) * page_size + entry_idx;
  kv_param[k_param] = __floats2half2_rn(k_scale, k_scale * 8.0f);
}
```

### 4.6 V 的 Hadamard + Quant + 寫入 cache

V 的流程與 K 相同，只是寫到 `k_or_v = 1` 的 plane。見 `attention_fusion/attention_fusion.hip:139`：

```cpp
hadamard_inplace(v_smem, kHeadDim);
float v_rot = v_smem[tid] * rsqrtf(static_cast<float>(kHeadDim));
v_smem[tid] = v_rot;
float v_scale = reduce_amax(v_rot, reduce);
```

V offset 在 `attention_fusion/attention_fusion.hip:145`：

```cpp
size_t v_elem = ((((page_idx * num_layers + layer_idx) * 2 + 1) * num_heads + head) * page_size + entry_idx) *
                (kHeadDim / 2) + (tid >> 1);
```

這裡的 `* 2 + 1` 就是 V plane。

## 5. K2：FlashInfer INT4 Paged KV Decode

### 5.1 Python 入口

在 `quarot_model_integration.py:97`：

```python
def decode_attention(self, query: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(query)
    flashinfer_hip.batch_decode_i4(
        out,
        query.contiguous(),
        self.kv_data,
        self.kv_param,
        ...
    )
    return out
```

K2 吃的是 K1 寫好的 `kv_data` / `kv_param`。

### 5.2 Wrapper 入口

`batch_decode_i4` 在 `flashinfer.hip:423`：

```cpp
void batch_decode_i4(
    torch::Tensor o, torch::Tensor q, torch::Tensor kv_data,
    torch::Tensor kv_param, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor last_page_offset, int num_layers, int layer_idx,
    int num_heads, int page_size, int batch_size)
```

它檢查 `kv_data` 是 CUDA/HIP tensor 且 dtype 是 `uint8`，然後呼叫：

```cpp
FlashInferBatchDecodeKernel_i4<128>(...)
```

### 5.3 FlashInfer typed paged KV view

在 `flashinfer.hip:13`：

```cpp
template <int head_dim>
void FlashInferBatchDecodeKernel_i4(...)
```

關鍵型別：

```cpp
using DTypeIn = flashinfer::quant::__precision__s4;
using DTypeInQ = __half;
using DTypeOut = __half;
```

建立 paged KV view，見 `flashinfer.hip:27`：

```cpp
flashinfer::paged_kv_t<DTypeIn, int32_t> paged_kv(
    num_layers, layer_idx, num_heads, page_size, head_dim, batch_size,
    (DTypeIn*)kv_data, kv_param, kv_indptr, kv_indicies, last_page_offset);
```

這行把 K1 的 raw `uint8_t` packed cache 解讀成 FlashInfer 的 signed INT4 KV cache。

### 5.4 K2 Kernel Launch

launch 在 `flashinfer.hip:53`：

```cpp
hipLaunchKernelGGL((flashinfer::BatchDecodeWithPagedKVCacheKernel<
    rotary_mode, norm_on_the_fly, vec_size, bdx, bdy, FoldFactor,
    DTypeInQ, DTypeIn, DTypeOut, int32_t>),
    dim3(nblks), dim3(nthrs), 0, 0,
    q, paged_kv, o, sm_scale, rope_inv_scale, rope_inv_theta);
```

K2 wrapper 自己沒有手寫 attention score loop。真正的工作在 FlashInfer template kernel 裡完成，包括：

- 從 paged KV cache 找 token pages。
- INT4 dequant。
- QK score accumulation。
- softmax / normalization。
- 對 V 做 weighted reduction。
- 寫出 FP16 attention output。

### 5.5 為什麼 K2 設 RotaryMode::kNone

見 `flashinfer.hip:37`：

```cpp
constexpr bool norm_on_the_fly = false;
constexpr auto rotary_mode = flashinfer::RotaryMode::kNone;
```

原因是 K1 已經可以在 append 前對 K 做 optional RoPE。K2 直接消費 cache 裡的 rotated / quantized K/V，不再重新做 RoPE。

### 5.6 GQA 版本

GQA 版本在 `flashinfer.hip:59`：

```cpp
void FlashInferBatchDecodeKernel_i4_gqa(...)
```

差異：

- cache 用 `num_kv_heads` 建立。
- launch grid 用 `num_q_heads`。
- FlashInfer kernel 會把 Q head 對應到 KV head group。

Python-facing wrapper 是 `batch_decode_i4_gqa`，見 `flashinfer.hip:436`。

## 6. K3：Attention Output Hadamard + INT4 Quant

### 6.1 Python 入口

在 `quarot_model_integration.py:117`：

```python
def quantize_attention(self, attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return quantize_attention_output(
        attention.reshape(self.batch_size, self.num_heads * self.head_dim),
        backend=self.k3_backend,
    )
```

`num_heads * head_dim` 必須等於 4096，檢查在 `quarot_model_integration.py:62`。

Python helper 在 `attention_fusion/quarot_attention_fusion.py:99`：

```python
def quantize_attention_output(attention_out: torch.Tensor, backend: str = "current"):
    if backend == "current":
        return attention_fusion_hip.output_had_quant(attention_out)
```

### 6.2 Kernel 入口與 Launch

current K3 kernel 在 `attention_fusion/attention_fusion.hip:157`：

```cpp
__global__ void output_had_quant_kernel(
    const half* __restrict__ attention_out,
    uint8_t* __restrict__ packed,
    half* __restrict__ scales,
    int rows)
```

launch 在 `attention_fusion/attention_fusion.hip:490`：

```cpp
dim3 grid(rows, kOutDim / kOutGroup);
dim3 block(kOutGroup);
size_t smem = static_cast<size_t>(kOutGroup) * 2 * sizeof(float);
```

常數在 `attention_fusion/attention_fusion.hip:17`：

```cpp
constexpr int kOutDim = 4096;
constexpr int kOutGroup = 256;
```

所以：

- 一列 attention output 是 4096。
- 切成 `4096 / 256 = 16` groups。
- 一個 block 負責一個 row 的一個 256-wide group。
- `threadIdx.x` 對應 group 內元素。

### 6.3 K3 Kernel 資料流

讀取 input，見 `attention_fusion/attention_fusion.hip:175`：

```cpp
float x = __half2float(attention_out[base + tid]);
values[tid] = x;
__syncthreads();
```

Hadamard + normalize，見 `attention_fusion/attention_fusion.hip:179`：

```cpp
hadamard_inplace(values, kOutGroup);
float rotated = values[tid] * rsqrtf(static_cast<float>(kOutGroup));
values[tid] = rotated;
```

Scale：

```cpp
float scale = reduce_amax(rotated, reduce);
if (tid == 0) {
  scales[row * (kOutDim / kOutGroup) + group] = __float2half(scale);
}
```

Pack：

```cpp
if ((tid & 1) == 0) {
  int q0 = max(-8, min(7, static_cast<int>(nearbyintf(values[tid] / scale))));
  int q1 = max(-8, min(7, static_cast<int>(nearbyintf(values[tid + 1] / scale))));
  packed[(row * kOutDim + group * kOutGroup + tid) >> 1] = pack_s4_pair(q0, q1);
}
```

輸出 shape：

- input：`[rows, 4096]`
- packed：`[rows, 2048]`
- scales：`[rows, 16]`

### 6.4 K3 Hadacore256

Hadacore256 kernel 在 `attention_fusion/attention_fusion.hip:194`：

```cpp
__global__ void output_had_quant_hadacore256_kernel(...)
```

它和 current K3 做同樣語意：

```text
256-wide group -> H256 -> amax -> scale -> INT4 pack
```

差異是：

- block 是 32 threads。
- 每個 lane 處理多個 group element。
- 使用 `quarot_hadamard_wmma::h256_f16` 做 H256。

### 6.5 K3 Hadacore4096 Experimental

experimental kernel 在 `attention_fusion/attention_fusion.hip:270`：

```cpp
__global__ void output_had_quant_hadacore4096_experimental_kernel(...)
```

這個版本語意和 current K3 不完全相同：

- current K3 是 16 個獨立 H256 block。
- experimental K3 是 full 4096-wide Hadamard。
- 實作上先做 16 個 local H256，再用 H16 混合 16 個 chunks。

因此它是 experimental backend，不應直接視為 current K3 的 drop-in replacement。

## 7. Hadacore / WMMA Hadamard 的角色

`hadamard_wmma.cuh` 提供 device-level Hadamard helper。它不是 standalone PyTorch extension，而是被 FFN/K3 fused kernel 直接 include：

```cpp
#include "../hadamard_wmma.cuh"
```

這樣做的原因：

- standalone extension 會重新引入額外 kernel launch。
- fused kernel 需要把 activation/Hadamard/quant/pack 留在同一個 kernel 內。
- Hadacore helper 只負責加速 Hadamard microkernel，不負責整個 QuaRot pipeline。

目前使用位置：

- FFN Hadacore256：`ffn_fusion/ffn_fusion.hip:128`
- K3 Hadacore256：`attention_fusion/attention_fusion.hip:217`
- K3 Hadacore4096 experimental：`attention_fusion/attention_fusion.hip:290`

## 8. Trace Code 建議路線

如果要從 Python trace 到 kernel，建議照這個順序：

1. `quarot_model_integration.py:138`
   看 `decode_step()` 如何串 K1/K2/K3/FFN。

2. `attention_fusion/quarot_attention_fusion.py:58`
   看 K1 Python helper 如何整理 metadata 並呼叫 HIP extension。

3. `attention_fusion/attention_fusion.hip:70`
   看 K1 kernel 如何定位 paged cache、處理 RoPE、Hadamard、quant、pack。

4. `flashinfer.hip:423`
   看 K2 Python binding 進來後如何呼叫 `FlashInferBatchDecodeKernel_i4<128>`。

5. `flashinfer.hip:13`
   看 K2 如何建立 `paged_kv_t` 並 launch FlashInfer template kernel。

6. `attention_fusion/quarot_attention_fusion.py:99`
   看 K3 helper 如何選 backend。

7. `attention_fusion/attention_fusion.hip:157`
   看 current K3 grouped H256 + INT4 quant。

8. `ffn_fusion/ffn_fusion.hip:230`
   看 FFN extension 如何檢查 shape、配置 output、根據 group size dispatch。

9. `ffn_fusion/ffn_fusion.hip:29`
   看 FFN fused kernel 完整資料流。

## 9. 常見實作陷阱

### 9.1 Scale shape 不同

FFN/K3 的 `scales` 是每個 group 一個 scalar half：

```text
FFN scales: [..., cols / group_size]
K3 scales: [rows, 16]
```

K1 的 `kv_param` 則是每個 K/V head token 一個 `half2(scale, scale * 8)`：

```text
kv_param[page, layer, k_or_v, head, token_slot, 2]
```

### 9.2 K1 的 K/V plane offset

K plane：

```cpp
((page_idx * num_layers + layer_idx) * 2 * num_heads + head)
```

V plane：

```cpp
(((page_idx * num_layers + layer_idx) * 2 + 1) * num_heads + head)
```

兩者差在 K/V plane 的 index。trace cache mismatch 時，這是第一個要檢查的地方。

### 9.3 RoPE 不能重複做

K1 有 `apply_rope_to_k`。如果 upstream 已經做過 RoPE，這裡就要設 `false`。否則 K 會被旋轉兩次，K2 decode 結果會錯。

### 9.4 Hadamard 語意要分清楚

current K3 是 grouped H256：

```text
[4096] -> 16 groups of [256] -> each group independent H256
```

experimental K3 是 full H4096：

```text
[4096] -> full H4096 mixing
```

兩者不是同一個 rotation。

### 9.5 `__syncthreads()` 不能省

Hadamard butterfly 每一輪都依賴上一輪 shared memory 結果。少一個 barrier 可能會在小測資偶爾過，但在真實 GPU scheduling 下產生 nondeterministic mismatch。

## 10. Kernel Fusion 為什麼有效

以 FFN/K1/K3 這類 rotation + quant path 來說，unfused 寫法通常需要：

```text
load FP16 tensor
write activation intermediate
read intermediate
write Hadamard intermediate
read Hadamard output
write quantized tensor
pack / copy / reshape
```

fused kernel 則變成：

```text
load FP16 input once
compute activation / RoPE / Hadamard in registers + shared memory
compute scale inside block
write packed INT4 + scale once
```

主要收益來自：

- 減少 kernel launch 次數。
- 減少 global memory intermediate tensor。
- Hadamard intermediate 留在 shared memory。
- pack 和 scale 寫出格式直接符合下一個 fused block 或 FlashInfer decode。

K2 則是另一種 fusion：它不只是 pack/quant 的 fusion，而是直接讓 FlashInfer decode kernel 在 attention decode 時讀 INT4 paged KV cache，避免先把整個 cache dequant 回 FP16 再做 attention。

## 11. 最小心智模型

可以把四個 fused block 記成：

```text
K1:
  current K/V FP16
  -> optional RoPE on K
  -> per-head H128
  -> per-head dynamic INT4
  -> paged INT4 KV cache

K2:
  Q FP16 + paged INT4 KV cache
  -> FlashInfer INT4 decode
  -> attention output FP16

K3:
  attention output FP16 [4096]
  -> 16 x H256
  -> per-group dynamic INT4
  -> packed attention output + scales

FFN:
  gate/up FP16
  -> SiLU(gate) * up
  -> per-group Hadamard
  -> per-group dynamic INT4
  -> packed FFN intermediate + scales
```

trace code 時只要一直確認三件事，通常就能定位大部分問題：

1. 這個 kernel 的 block/thread 對應哪個 tensor 維度？
2. 這個 scale 是 scalar per group，還是 `half2(scale, scale * 8)`？
3. 這個 packed byte 的低/高 nibble 對應哪兩個 logical elements？
