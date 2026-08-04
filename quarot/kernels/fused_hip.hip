#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include <torch/extension.h>

#include <fused.h>

namespace {

constexpr int kHeadDim = 128;
constexpr int kThreads = 256;
constexpr int kMaxFusedWidth = 8192;
constexpr int kMaxGeneralFfnWidth = 14336;

__device__ __forceinline__ uint8_t pack_s4_pair(int lo, int hi) {
  // QuaRot GEMM consumes two's-complement signed nibbles, low element first.
  return static_cast<uint8_t>((lo & 0x0f) | ((hi & 0x0f) << 4));
}

__device__ __forceinline__ uint8_t pack_u4_pair(int lo, int hi) {
  return static_cast<uint8_t>((lo & 0x0f) | ((hi & 0x0f) << 4));
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float reduce_max(float value, float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
    __syncthreads();
  }
  return scratch[0];
}

// Adapted from third-party/hadacore's HIP N=32/64 small-warp path.
// gfx12 executes wave32; keeping the transform in registers removes the six
// workgroup-wide barriers and repeated LDS traffic of the scalar butterfly.
__device__ __forceinline__ float hadacore_shfl_xor_f32(float value, int mask) {
  return __shfl_xor_sync(0xffffffffULL, value, mask, 32);
}

__device__ __forceinline__ void hadacore_h64_regs(float values[2], int lane) {
  for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const float a = values[r];
      const float b = hadacore_shfl_xor_f32(a, mask);
      values[r] = (lane & mask) == 0 ? a + b : b - a;
    }
  }
  const float a = values[0];
  const float b = values[1];
  values[0] = a + b;
  values[1] = a - b;
}

__device__ __forceinline__ void hadamard(float* values, int width) {
  for (int stride = 1; stride < width; stride <<= 1) {
    for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
      int low = ((pair / stride) * (stride << 1)) + (pair & (stride - 1));
      float a = values[low];
      float b = values[low + stride];
      values[low] = a + b;
      values[low + stride] = a - b;
    }
    __syncthreads();
  }
}

__global__ void fused_append_kv_i4_kernel(
    const half* key, const half* value, uint8_t* kv_data, half2* kv_param,
    const int32_t* kv_indptr, const int32_t* kv_indices,
    const int32_t* last_page_offset, int num_layers, int layer_idx,
    int num_heads, int page_size, int batch_size) {
  __shared__ float k_values[kHeadDim];
  __shared__ float v_values[kHeadDim];
  __shared__ float scratch[kHeadDim];
  const int tid = threadIdx.x;
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  if (batch >= batch_size || head >= num_heads || tid >= kHeadDim) return;

  const int seq_len = (kv_indptr[batch + 1] - kv_indptr[batch] - 1) * page_size + last_page_offset[batch];
  if (seq_len <= 0) return;
  const int page = kv_indices[kv_indptr[batch] + (seq_len - 1) / page_size];
  const int slot = (seq_len - 1) % page_size;
  const int input = (batch * num_heads + head) * kHeadDim + tid;
  k_values[tid] = __half2float(key[input]);
  v_values[tid] = __half2float(value[input]);
  __syncthreads();

  hadamard(k_values, kHeadDim);
  float kx = k_values[tid] * rsqrtf(float(kHeadDim));
  k_values[tid] = kx;
  float kmax = reduce_max(kx, scratch);
  __syncthreads();
  float kmin = -reduce_max(-kx, scratch);
  half kscale_h = __float2half(fmaxf((kmax - kmin) / 15.0f, 1.0e-5f));
  half kzero_h = __float2half(-kmin);
  float kscale = __half2float(kscale_h);
  float kzero = __half2float(kzero_h);
  if ((tid & 1) == 0) {
    int q0 = max(0, min(15, static_cast<int>(nearbyintf((k_values[tid] + kzero) / kscale))));
    int q1 = max(0, min(15, static_cast<int>(nearbyintf((k_values[tid + 1] + kzero) / kscale))));
    size_t index = (((page * num_layers + layer_idx) * 2 * num_heads + head) * page_size + slot) * (kHeadDim / 2) + (tid >> 1);
    kv_data[index] = pack_u4_pair(q0, q1);
  }
  if (tid == 0) {
    size_t index = ((page * num_layers + layer_idx) * 2 * num_heads + head) * page_size + slot;
    kv_param[index] = __halves2half2(kscale_h, kzero_h);
  }
  __syncthreads();

  hadamard(v_values, kHeadDim);
  float vx = v_values[tid] * rsqrtf(float(kHeadDim));
  v_values[tid] = vx;
  float vmax = reduce_max(vx, scratch);
  __syncthreads();
  float vmin = -reduce_max(-vx, scratch);
  half vscale_h = __float2half(fmaxf((vmax - vmin) / 15.0f, 1.0e-5f));
  half vzero_h = __float2half(-vmin);
  float vscale = __half2float(vscale_h);
  float vzero = __half2float(vzero_h);
  if ((tid & 1) == 0) {
    int q0 = max(0, min(15, static_cast<int>(nearbyintf((v_values[tid] + vzero) / vscale))));
    int q1 = max(0, min(15, static_cast<int>(nearbyintf((v_values[tid + 1] + vzero) / vscale))));
    size_t index = ((((page * num_layers + layer_idx) * 2 + 1) * num_heads + head) * page_size + slot) * (kHeadDim / 2) + (tid >> 1);
    kv_data[index] = pack_u4_pair(q0, q1);
  }
  if (tid == 0) {
    size_t index = (((page * num_layers + layer_idx) * 2 + 1) * num_heads + head) * page_size + slot;
    kv_param[index] = __halves2half2(vscale_h, vzero_h);
  }
}

__global__ void fused_attention_hadamard_quant_kernel(
    const half* attention, uint8_t* packed, half* scales, int rows,
    int num_heads, int head_dim) {
  __shared__ float values[kMaxFusedWidth];
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  const int width = num_heads * head_dim;
  if (row >= rows) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const int d = i / num_heads;
    const int h = i % num_heads;
    values[i] = __half2float(attention[(row * num_heads + h) * head_dim + d]);
  }
  __syncthreads();
  // values is [head_dim, num_heads], matching OnlineHadamard after transpose.
  for (int d = 0; d < head_dim; ++d) {
    float* group = values + d * num_heads;
    for (int stride = 1; stride < num_heads; stride <<= 1) {
      for (int pair = threadIdx.x; pair < num_heads / 2; pair += blockDim.x) {
        int low = ((pair / stride) * (stride << 1)) + (pair & (stride - 1));
        float a = group[low];
        float b = group[low + stride];
        group[low] = a + b;
        group[low + stride] = a - b;
      }
      __syncthreads();
    }
  }
  const float norm = rsqrtf(float(num_heads));
  float local_max = 0.0f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    values[i] *= norm;
    local_max = fmaxf(local_max, fabsf(values[i]));
  }
  half scale_h = __float2half(fmaxf(reduce_max(local_max, scratch) / 7.0f, 6.103515625e-5f));
  float scale = __half2float(scale_h);
  if (threadIdx.x == 0) scales[row] = scale_h;
  for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
    int out0 = pair << 1;
    int out1 = out0 + 1;
    int h0 = out0 / head_dim, d0 = out0 % head_dim;
    int h1 = out1 / head_dim, d1 = out1 % head_dim;
    int q0 = max(-8, min(7, static_cast<int>(nearbyintf(values[d0 * num_heads + h0] / scale))));
    int q1 = max(-8, min(7, static_cast<int>(nearbyintf(values[d1 * num_heads + h1] / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}

__global__ void fused_ffn_silu_hadamard_quant_kernel(
    const half* gate, const half* up, uint8_t* packed, half* scales,
    int rows, int width) {
  __shared__ float values[kMaxFusedWidth];
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x) values[i] = silu(__half2float(gate[row * width + i])) * __half2float(up[row * width + i]);
  __syncthreads();
  hadamard(values, width);
  float local_max = 0.0f;
  const float norm = rsqrtf(float(width));
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    values[i] *= norm;
    local_max = fmaxf(local_max, fabsf(values[i]));
  }
  half scale_h = __float2half(fmaxf(reduce_max(local_max, scratch) / 7.0f, 6.103515625e-5f));
  float scale = __half2float(scale_h);
  if (threadIdx.x == 0) scales[row] = scale_h;
  for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
    int q0 = max(-8, min(7, static_cast<int>(nearbyintf(values[pair * 2] / scale))));
    int q1 = max(-8, min(7, static_cast<int>(nearbyintf(values[pair * 2 + 1] / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}

// For width=K*P, QuaRot applies a power-of-two H_P followed by its
// fixed H_K remainder matrix. Recompute H_K for quantization to avoid a second
// width-sized LDS buffer (which exceeds 64 KiB for Llama-2 7B's width 11008).
__global__ void fused_ffn_silu_hadamard_quant_general_kernel(
    const half* gate, const half* up, const half* hadamard, uint8_t* packed,
    half* scales, int rows, int width, int remainder, int inner) {
  extern __shared__ float values[];
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  if (row >= rows) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    values[i] = silu(__half2float(gate[row * width + i])) *
                __half2float(up[row * width + i]);
  }
  __syncthreads();

  if (inner == 64) {
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int waves = blockDim.x >> 5;
    for (int group = wave; group < remainder; group += waves) {
      float registers[2] = {values[group * 64 + lane],
                            values[group * 64 + 32 + lane]};
      hadacore_h64_regs(registers, lane);
      values[group * 64 + lane] = registers[0];
      values[group * 64 + 32 + lane] = registers[1];
    }
    __syncthreads();
  } else {
    for (int stride = 1; stride < inner; stride <<= 1) {
      const int pairs_per_group = inner >> 1;
      for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
        const int group = pair / pairs_per_group;
        const int local_pair = pair - group * pairs_per_group;
        const int low = group * inner +
            (local_pair / stride) * (stride << 1) +
            (local_pair & (stride - 1));
        const float a = values[low];
        const float b = values[low + stride];
        values[low] = a + b;
        values[low + stride] = a - b;
      }
      __syncthreads();
    }
  }

  constexpr int kMaxPairsPerThread =
      (kMaxGeneralFfnWidth / 2 + kThreads - 1) / kThreads;
  float transformed[kMaxPairsPerThread * 2];
  const float norm = rsqrtf(float(width));
  float local_max = 0.0f;
  int owned_pairs = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned_pairs) {
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      const int out = pair * 2 + which;
      const int out_k = out / inner;
      const int p = out - out_k * inner;
      float value = 0.0f;
      for (int in_k = 0; in_k < remainder; ++in_k) {
        value += __half2float(hadamard[out_k * remainder + in_k]) *
                 values[in_k * inner + p];
      }
      value *= norm;
      transformed[owned_pairs * 2 + which] = value;
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  half scale_h = __float2half(fmaxf(reduce_max(local_max, scratch) / 7.0f,
                                    6.103515625e-5f));
  const float scale = __half2float(scale_h);
  if (threadIdx.x == 0) scales[row] = scale_h;

  owned_pairs = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned_pairs) {
    const int q0 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned_pairs * 2] / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned_pairs * 2 + 1] / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}

void check_cuda_half_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA/HIP tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

bool power_of_two(int64_t x) { return x > 0 && (x & (x - 1)) == 0; }

}  // namespace

void fused_append_kv_i4(
    torch::Tensor kv_data, torch::Tensor kv_param, torch::Tensor kv_indptr,
    torch::Tensor kv_indices, torch::Tensor last_page_offset, torch::Tensor key,
    torch::Tensor value, int64_t num_layers, int64_t layer_idx,
    int64_t num_heads, int64_t page_size, int64_t batch_size) {
  check_cuda_half_contiguous(key, "key");
  check_cuda_half_contiguous(value, "value");
  TORCH_CHECK(kv_data.is_cuda() && kv_data.is_contiguous() && kv_data.scalar_type() == torch::kUInt8, "kv_data must be a contiguous uint8 CUDA/HIP tensor");
  TORCH_CHECK(kv_param.is_cuda() && kv_param.is_contiguous() && kv_param.scalar_type() == torch::kFloat16, "kv_param must be a contiguous float16 CUDA/HIP tensor");
  TORCH_CHECK(kv_indptr.is_cuda() && kv_indices.is_cuda() && last_page_offset.is_cuda(), "paged-cache metadata must be CUDA/HIP tensors");
  TORCH_CHECK(kv_indptr.is_contiguous() && kv_indices.is_contiguous() && last_page_offset.is_contiguous(), "paged-cache metadata must be contiguous");
  TORCH_CHECK(kv_indptr.scalar_type() == torch::kInt32 && kv_indices.scalar_type() == torch::kInt32 && last_page_offset.scalar_type() == torch::kInt32, "paged-cache metadata must be int32");
  TORCH_CHECK(kv_data.get_device() == key.get_device() && kv_param.get_device() == key.get_device() &&
              kv_indptr.get_device() == key.get_device() && kv_indices.get_device() == key.get_device() &&
              last_page_offset.get_device() == key.get_device() && value.get_device() == key.get_device(),
              "all fused KV append tensors must be on the same device");
  TORCH_CHECK(kv_indptr.numel() == batch_size + 1 && last_page_offset.numel() == batch_size,
              "paged-cache metadata lengths do not match batch_size");
  TORCH_CHECK(num_layers > 0 && num_heads > 0 && page_size > 0 && batch_size > 0,
              "cache dimensions must be positive");
  TORCH_CHECK(key.sizes() == value.sizes() && key.dim() == 3 && key.size(0) == batch_size && key.size(1) == num_heads && key.size(2) == kHeadDim, "key/value must have shape [batch, heads, 128]");
  TORCH_CHECK(layer_idx >= 0 && layer_idx < num_layers, "layer_idx is outside the cache");
  at::hip::HIPGuard guard{static_cast<char>(key.get_device())};
  hipLaunchKernelGGL(fused_append_kv_i4_kernel, dim3(batch_size, num_heads), dim3(kHeadDim), 0,
      at::hip::getCurrentHIPStream().stream(), reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value.data_ptr<at::Half>()), kv_data.data_ptr<uint8_t>(),
      reinterpret_cast<half2*>(kv_param.data_ptr<at::Half>()), kv_indptr.data_ptr<int32_t>(),
      kv_indices.data_ptr<int32_t>(), last_page_offset.data_ptr<int32_t>(), num_layers,
      layer_idx, num_heads, page_size, batch_size);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "fused_append_kv_i4 launch failed");
}

std::vector<torch::Tensor> fused_attention_hadamard_quant(torch::Tensor attention, int64_t num_heads) {
  check_cuda_half_contiguous(attention, "attention");
  TORCH_CHECK(attention.dim() == 4, "attention must have shape [batch, sequence, heads, head_dim]");
  TORCH_CHECK(attention.size(2) == num_heads && power_of_two(num_heads), "num_heads must be a positive power of two");
  const int64_t head_dim = attention.size(3);
  const int64_t width = num_heads * head_dim;
  TORCH_CHECK(width <= kMaxFusedWidth && width % 2 == 0, "fused attention supports even hidden sizes up to ", kMaxFusedWidth);
  const int64_t rows = attention.numel() / width;
  TORCH_CHECK(rows > 0, "fused attention does not support empty tensors");
  auto packed = torch::empty({attention.size(0), attention.size(1), width / 2}, attention.options().dtype(torch::kUInt8));
  auto scales = torch::empty({attention.size(0), attention.size(1), 1}, attention.options());
  at::hip::HIPGuard guard{static_cast<char>(attention.get_device())};
  hipLaunchKernelGGL(fused_attention_hadamard_quant_kernel, dim3(rows), dim3(kThreads), 0,
      at::hip::getCurrentHIPStream().stream(), reinterpret_cast<const half*>(attention.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(), reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, num_heads, head_dim);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "fused_attention_hadamard_quant launch failed");
  return {packed, scales};
}

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant(torch::Tensor gate, torch::Tensor up) {
  check_cuda_half_contiguous(gate, "gate");
  check_cuda_half_contiguous(up, "up");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.dim() >= 1, "gate and up must have identical non-empty shapes");
  const int64_t width = gate.size(-1);
  TORCH_CHECK(power_of_two(width) && width <= kMaxFusedWidth && width % 2 == 0, "fused FFN supports power-of-two widths up to ", kMaxFusedWidth);
  const int64_t rows = gate.numel() / width;
  TORCH_CHECK(rows > 0, "fused FFN does not support empty tensors");
  auto packed_shape = gate.sizes().vec(); packed_shape.back() = width / 2;
  auto scale_shape = gate.sizes().vec(); scale_shape.back() = 1;
  auto packed = torch::empty(packed_shape, gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, gate.options());
  at::hip::HIPGuard guard{static_cast<char>(gate.get_device())};
  hipLaunchKernelGGL(fused_ffn_silu_hadamard_quant_kernel, dim3(rows), dim3(kThreads), 0,
      at::hip::getCurrentHIPStream().stream(), reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()), packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, width);
  TORCH_CHECK(hipGetLastError() == hipSuccess, "fused_ffn_silu_hadamard_quant launch failed");
  return {packed, scales};
}


std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_general(
    torch::Tensor gate, torch::Tensor up, torch::Tensor hadamard) {
  check_cuda_half_contiguous(gate, "gate");
  check_cuda_half_contiguous(up, "up");
  check_cuda_half_contiguous(hadamard, "hadamard");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.dim() >= 1,
              "gate and up must have identical non-empty shapes");
  TORCH_CHECK(hadamard.dim() == 2 && hadamard.size(0) == hadamard.size(1),
              "hadamard must be a square matrix");
  TORCH_CHECK(gate.get_device() == hadamard.get_device(),
              "gate, up, and hadamard must be on the same device");
  const int64_t width = gate.size(-1);
  const int64_t remainder = hadamard.size(0);
  TORCH_CHECK(width > 0 && width <= kMaxGeneralFfnWidth &&
                  width % remainder == 0 && width % 2 == 0,
              "width must be a positive even multiple of the Hadamard order and at most ",
              kMaxGeneralFfnWidth);
  const int64_t inner = width / remainder;
  TORCH_CHECK(power_of_two(inner),
              "width / Hadamard order must be a power of two");
  const int64_t rows = gate.numel() / width;
  TORCH_CHECK(rows > 0, "fused FFN does not support empty tensors");
  auto packed_shape = gate.sizes().vec(); packed_shape.back() = width / 2;
  auto scale_shape = gate.sizes().vec(); scale_shape.back() = 1;
  auto packed = torch::empty(packed_shape, gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, gate.options());
  at::hip::HIPGuard guard{static_cast<char>(gate.get_device())};
  hipLaunchKernelGGL(fused_ffn_silu_hadamard_quant_general_kernel,
      dim3(rows), dim3(kThreads), width * sizeof(float),
      at::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(hadamard.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, width,
      remainder, inner);
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "fused_ffn_silu_hadamard_quant_general launch failed");
  return {packed, scales};
}
