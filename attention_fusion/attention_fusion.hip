#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

#define HIP_ASSERT(x) TORCH_CHECK((x) == hipSuccess, "HIP error: ", hipGetErrorString(x))

namespace {

constexpr int kHeadDim = 128;
constexpr int kOutDim = 4096;
constexpr int kOutGroup = 256;

__device__ __forceinline__ uint8_t pack_s4_pair(int lo, int hi) {
  uint8_t ulo = static_cast<uint8_t>(lo + 8) & 0x0f;
  uint8_t uhi = static_cast<uint8_t>(hi + 8) & 0x0f;
  return static_cast<uint8_t>(ulo | (uhi << 4));
}

__device__ __forceinline__ void hadamard_inplace(float* data, int n) {
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
}

__device__ __forceinline__ void rope_pair(float& x0, float& x1, int pair_idx, int pos, float inv_theta) {
  float freq = powf(inv_theta, static_cast<float>(2 * pair_idx) / static_cast<float>(kHeadDim));
  float angle = static_cast<float>(pos) * freq;
  float s = sinf(angle);
  float c = cosf(angle);
  float y0 = x0 * c - x1 * s;
  float y1 = x0 * s + x1 * c;
  x0 = y0;
  x1 = y1;
}

__device__ __forceinline__ float reduce_amax(float x, float* smem) {
  int tid = threadIdx.x;
  smem[tid] = fabsf(x);
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }
  return fmaxf(smem[0] / 7.0f, 1.0e-8f);
}

__global__ void append_kv_had_quant_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    uint8_t* __restrict__ kv_data,
    half2* __restrict__ kv_param,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ last_page_offset,
    int num_layers,
    int layer_idx,
    int num_heads,
    int page_size,
    int batch_size,
    bool apply_rope_to_k,
    float rope_inv_theta) {
  __shared__ float k_smem[kHeadDim];
  __shared__ float v_smem[kHeadDim];
  __shared__ float reduce[kHeadDim];

  int tid = threadIdx.x;
  int batch = blockIdx.x;
  int head = blockIdx.y;
  if (batch >= batch_size || head >= num_heads) return;

  int seq_len = (kv_indptr[batch + 1] - kv_indptr[batch] - 1) * page_size + last_page_offset[batch];
  int page_idx = kv_indices[kv_indptr[batch] + (seq_len - 1) / page_size];
  int entry_idx = (seq_len - 1) % page_size;
  int input_offset = (batch * num_heads + head) * kHeadDim + tid;

  float kval = __half2float(key[input_offset]);
  if (apply_rope_to_k) {
    int pair_idx = tid >> 1;
    int pair_base = (batch * num_heads + head) * kHeadDim + (pair_idx << 1);
    float x0 = __half2float(key[pair_base]);
    float x1 = __half2float(key[pair_base + 1]);
    rope_pair(x0, x1, pair_idx, seq_len - 1, rope_inv_theta);
    kval = (tid & 1) ? x1 : x0;
  }
  k_smem[tid] = kval;
  v_smem[tid] = __half2float(value[input_offset]);
  __syncthreads();

  hadamard_inplace(k_smem, kHeadDim);
  float k_rot = k_smem[tid] * rsqrtf(static_cast<float>(kHeadDim));
  k_smem[tid] = k_rot;
  float k_scale = reduce_amax(k_rot, reduce);
  int kq = max(-8, min(7, static_cast<int>(nearbyintf(k_rot / k_scale))));
  if ((tid & 1) == 0) {
    int kq1 = max(-8, min(7, static_cast<int>(nearbyintf(k_smem[tid + 1] / k_scale))));
    size_t k_elem = (((page_idx * num_layers + layer_idx) * 2 * num_heads + head) * page_size + entry_idx) *
                    (kHeadDim / 2) + (tid >> 1);
    kv_data[k_elem] = pack_s4_pair(kq, kq1);
  }
  if (tid == 0) {
    size_t k_param = ((page_idx * num_layers + layer_idx) * 2 * num_heads + head) * page_size + entry_idx;
    kv_param[k_param] = __floats2half2_rn(k_scale, k_scale * 8.0f);
  }
  __syncthreads();

  hadamard_inplace(v_smem, kHeadDim);
  float v_rot = v_smem[tid] * rsqrtf(static_cast<float>(kHeadDim));
  v_smem[tid] = v_rot;
  float v_scale = reduce_amax(v_rot, reduce);
  int vq = max(-8, min(7, static_cast<int>(nearbyintf(v_rot / v_scale))));
  if ((tid & 1) == 0) {
    int vq1 = max(-8, min(7, static_cast<int>(nearbyintf(v_smem[tid + 1] / v_scale))));
    size_t v_elem = ((((page_idx * num_layers + layer_idx) * 2 + 1) * num_heads + head) * page_size + entry_idx) *
                    (kHeadDim / 2) + (tid >> 1);
    kv_data[v_elem] = pack_s4_pair(vq, vq1);
  }
  if (tid == 0) {
    size_t v_param = (((page_idx * num_layers + layer_idx) * 2 + 1) * num_heads + head) * page_size + entry_idx;
    kv_param[v_param] = __floats2half2_rn(v_scale, v_scale * 8.0f);
  }
}

__global__ void output_had_quant_kernel(
    const half* __restrict__ attention_out,
    uint8_t* __restrict__ packed,
    half* __restrict__ scales,
    int rows) {
  extern __shared__ float smem[];
  float* values = smem;
  float* reduce = smem + kOutGroup;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int group = blockIdx.y;
  int base = row * kOutDim + group * kOutGroup;

  float x = __half2float(attention_out[base + tid]);
  values[tid] = x;
  __syncthreads();

  hadamard_inplace(values, kOutGroup);
  float rotated = values[tid] * rsqrtf(static_cast<float>(kOutGroup));
  values[tid] = rotated;
  float scale = reduce_amax(rotated, reduce);
  if (tid == 0) {
    scales[row * (kOutDim / kOutGroup) + group] = __float2half(scale);
  }
  if ((tid & 1) == 0) {
    int q0 = max(-8, min(7, static_cast<int>(nearbyintf(values[tid] / scale))));
    int q1 = max(-8, min(7, static_cast<int>(nearbyintf(values[tid + 1] / scale))));
    packed[(row * kOutDim + group * kOutGroup + tid) >> 1] = pack_s4_pair(q0, q1);
  }
}

int32_t* int32_ptr(torch::Tensor t) {
  TORCH_CHECK(t.is_cuda(), "index tensor must be on CUDA/HIP device");
  TORCH_CHECK(t.scalar_type() == torch::kInt32, "index tensor must be int32");
  return t.data_ptr<int32_t>();
}

std::vector<torch::Tensor> append_kv_had_quant(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    torch::Tensor last_page_offset,
    int64_t num_layers,
    int64_t layer_idx,
    int64_t num_heads,
    int64_t page_size,
    int64_t batch_size,
    int64_t max_num_pages,
    bool apply_rope_to_k,
    double rope_theta) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda(), "key/value must be CUDA/HIP tensors");
  TORCH_CHECK(key.scalar_type() == torch::kFloat16 && value.scalar_type() == torch::kFloat16,
              "key/value must be float16");
  TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), "key/value must be contiguous");
  TORCH_CHECK(key.numel() == batch_size * num_heads * kHeadDim, "key shape must be [batch, heads, 128]");
  TORCH_CHECK(value.numel() == key.numel(), "value shape must match key");

  auto kv_data = torch::empty(
      {max_num_pages, num_layers, 2, num_heads, page_size, kHeadDim / 2},
      key.options().dtype(torch::kUInt8));
  auto kv_param = torch::empty(
      {max_num_pages, num_layers, 2, num_heads, page_size, 2},
      key.options());

  at::hip::HIPGuard device_guard{static_cast<char>(key.get_device())};
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  dim3 grid(static_cast<unsigned int>(batch_size), static_cast<unsigned int>(num_heads));
  dim3 block(kHeadDim);
  hipLaunchKernelGGL(
      append_kv_had_quant_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value.data_ptr<at::Half>()),
      kv_data.data_ptr<uint8_t>(),
      reinterpret_cast<half2*>(kv_param.data_ptr<at::Half>()),
      int32_ptr(kv_indptr),
      int32_ptr(kv_indices),
      int32_ptr(last_page_offset),
      static_cast<int>(num_layers),
      static_cast<int>(layer_idx),
      static_cast<int>(num_heads),
      static_cast<int>(page_size),
      static_cast<int>(batch_size),
      apply_rope_to_k,
      static_cast<float>(1.0 / rope_theta));
  HIP_ASSERT(hipGetLastError());
  return {kv_data, kv_param};
}

void append_kv_had_quant_inplace(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor kv_data,
    torch::Tensor kv_param,
    torch::Tensor kv_indptr,
    torch::Tensor kv_indices,
    torch::Tensor last_page_offset,
    int64_t num_layers,
    int64_t layer_idx,
    int64_t num_heads,
    int64_t page_size,
    int64_t batch_size,
    bool apply_rope_to_k,
    double rope_theta) {
  TORCH_CHECK(key.is_cuda() && value.is_cuda(), "key/value must be CUDA/HIP tensors");
  TORCH_CHECK(kv_data.is_cuda() && kv_param.is_cuda(), "kv_data/kv_param must be CUDA/HIP tensors");
  TORCH_CHECK(key.scalar_type() == torch::kFloat16 && value.scalar_type() == torch::kFloat16,
              "key/value must be float16");
  TORCH_CHECK(kv_data.scalar_type() == torch::kUInt8, "kv_data must be uint8");
  TORCH_CHECK(kv_param.scalar_type() == torch::kFloat16, "kv_param must be float16");
  TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), "key/value must be contiguous");
  TORCH_CHECK(kv_data.is_contiguous() && kv_param.is_contiguous(), "kv_data/kv_param must be contiguous");
  TORCH_CHECK(key.numel() == batch_size * num_heads * kHeadDim, "key shape must be [batch, heads, 128]");
  TORCH_CHECK(value.numel() == key.numel(), "value shape must match key");

  at::hip::HIPGuard device_guard{static_cast<char>(key.get_device())};
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  dim3 grid(static_cast<unsigned int>(batch_size), static_cast<unsigned int>(num_heads));
  dim3 block(kHeadDim);
  hipLaunchKernelGGL(
      append_kv_had_quant_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value.data_ptr<at::Half>()),
      kv_data.data_ptr<uint8_t>(),
      reinterpret_cast<half2*>(kv_param.data_ptr<at::Half>()),
      int32_ptr(kv_indptr),
      int32_ptr(kv_indices),
      int32_ptr(last_page_offset),
      static_cast<int>(num_layers),
      static_cast<int>(layer_idx),
      static_cast<int>(num_heads),
      static_cast<int>(page_size),
      static_cast<int>(batch_size),
      apply_rope_to_k,
      static_cast<float>(1.0 / rope_theta));
  HIP_ASSERT(hipGetLastError());
}

std::vector<torch::Tensor> output_had_quant(torch::Tensor attention_out) {
  TORCH_CHECK(attention_out.is_cuda(), "attention_out must be a CUDA/HIP tensor");
  TORCH_CHECK(attention_out.scalar_type() == torch::kFloat16, "attention_out must be float16");
  TORCH_CHECK(attention_out.is_contiguous(), "attention_out must be contiguous");
  TORCH_CHECK(attention_out.size(-1) == kOutDim, "attention_out last dimension must be 4096");

  int rows = static_cast<int>(attention_out.numel() / kOutDim);
  auto packed_shape = attention_out.sizes().vec();
  packed_shape.back() = kOutDim / 2;
  auto scale_shape = attention_out.sizes().vec();
  scale_shape.back() = kOutDim / kOutGroup;
  auto packed = torch::empty(packed_shape, attention_out.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, attention_out.options());

  at::hip::HIPGuard device_guard{static_cast<char>(attention_out.get_device())};
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  dim3 grid(rows, kOutDim / kOutGroup);
  dim3 block(kOutGroup);
  size_t smem = static_cast<size_t>(kOutGroup) * 2 * sizeof(float);
  hipLaunchKernelGGL(
      output_had_quant_kernel,
      grid,
      block,
      smem,
      stream,
      reinterpret_cast<const half*>(attention_out.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
      rows);
  HIP_ASSERT(hipGetLastError());
  return {packed, scales};
}

void output_had_quant_inplace(torch::Tensor attention_out, torch::Tensor packed, torch::Tensor scales) {
  TORCH_CHECK(attention_out.is_cuda(), "attention_out must be a CUDA/HIP tensor");
  TORCH_CHECK(packed.is_cuda() && scales.is_cuda(), "packed/scales must be CUDA/HIP tensors");
  TORCH_CHECK(attention_out.scalar_type() == torch::kFloat16, "attention_out must be float16");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "packed must be uint8");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat16, "scales must be float16");
  TORCH_CHECK(attention_out.is_contiguous() && packed.is_contiguous() && scales.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(attention_out.size(-1) == kOutDim, "attention_out last dimension must be 4096");

  int rows = static_cast<int>(attention_out.numel() / kOutDim);
  at::hip::HIPGuard device_guard{static_cast<char>(attention_out.get_device())};
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();
  dim3 grid(rows, kOutDim / kOutGroup);
  dim3 block(kOutGroup);
  size_t smem = static_cast<size_t>(kOutGroup) * 2 * sizeof(float);
  hipLaunchKernelGGL(
      output_had_quant_kernel,
      grid,
      block,
      smem,
      stream,
      reinterpret_cast<const half*>(attention_out.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
      rows);
  HIP_ASSERT(hipGetLastError());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "append_kv_had_quant",
      &append_kv_had_quant,
      pybind11::arg("key"),
      pybind11::arg("value"),
      pybind11::arg("kv_indptr"),
      pybind11::arg("kv_indices"),
      pybind11::arg("last_page_offset"),
      pybind11::arg("num_layers"),
      pybind11::arg("layer_idx"),
      pybind11::arg("num_heads"),
      pybind11::arg("page_size"),
      pybind11::arg("batch_size"),
      pybind11::arg("max_num_pages"),
      pybind11::arg("apply_rope_to_k") = false,
      pybind11::arg("rope_theta") = 10000.0);
  m.def(
      "append_kv_had_quant_inplace",
      &append_kv_had_quant_inplace,
      pybind11::arg("key"),
      pybind11::arg("value"),
      pybind11::arg("kv_data"),
      pybind11::arg("kv_param"),
      pybind11::arg("kv_indptr"),
      pybind11::arg("kv_indices"),
      pybind11::arg("last_page_offset"),
      pybind11::arg("num_layers"),
      pybind11::arg("layer_idx"),
      pybind11::arg("num_heads"),
      pybind11::arg("page_size"),
      pybind11::arg("batch_size"),
      pybind11::arg("apply_rope_to_k") = false,
      pybind11::arg("rope_theta") = 10000.0);
  m.def("output_had_quant", &output_had_quant, pybind11::arg("attention_out"));
  m.def(
      "output_had_quant_inplace",
      &output_had_quant_inplace,
      pybind11::arg("attention_out"),
      pybind11::arg("packed"),
      pybind11::arg("scales"));
}
