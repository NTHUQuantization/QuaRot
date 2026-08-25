#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <type_traits>

#include <torch/extension.h>

#include <fused.h>
#include <hadamard_wmma.cuh>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxFusedWidth = 8192;
constexpr int kMaxGeneralFfnWidth = 14336;
// Qwen3-32B uses a 25 * 1024 transform. A float row needs 100 KiB,
// so widths above the single-block LDS limit use a tiled pipeline.
constexpr int kMaxLargeGeneralFfnWidth = 28672;

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

__global__ void fused_rmsnorm_quant_i4_kernel(
    const half* input, uint8_t* packed, half* scales, int rows, int cols,
    float eps, float clip_ratio) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float sum_scratch[kThreads];
  __shared__ float max_scratch[kThreads];
  float sum = 0.0f;
  float max_abs = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __half2float(input[row * cols + col]);
    sum = fmaf(value, value, sum);
    max_abs = fmaxf(max_abs, fabsf(value));
  }
  sum_scratch[threadIdx.x] = sum;
  max_scratch[threadIdx.x] = max_abs;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_scratch[threadIdx.x] += sum_scratch[threadIdx.x + stride];
      max_scratch[threadIdx.x] = fmaxf(
          max_scratch[threadIdx.x], max_scratch[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float inv_rms = rsqrtf(sum_scratch[0] / cols + eps);
  const float scale = fmaxf(
      max_scratch[0] * inv_rms * clip_ratio / 7.0f, 0x1.0p-14f);
  if (threadIdx.x == 0) scales[row] = __float2half(scale);
  for (int pair = threadIdx.x; pair < cols / 2; pair += blockDim.x) {
    const int col = pair * 2;
    int lo = __float2int_rn(
        __half2float(input[row * cols + col]) * inv_rms / scale);
    int hi = __float2int_rn(
        __half2float(input[row * cols + col + 1]) * inv_rms / scale);
    lo = max(-8, min(7, lo));
    hi = max(-8, min(7, hi));
    packed[row * (cols / 2) + pair] = pack_s4_pair(lo, hi);
  }
}

// Adapted from third-party/hadacore's HIP N=32/64 small-warp path.
// gfx12 executes wave32; keeping the transform in registers removes the six
// workgroup-wide barriers and repeated LDS traffic of the scalar butterfly.
__device__ __forceinline__ float hadacore_shfl_xor_f32(float value, int mask) {
  return __shfl_xor_sync(0xffffffffULL, value, mask, 32);
}

template <int Regs>
__device__ __forceinline__ void hadacore_warp_hadamard_regs(
    float values[2], int lane) {
  for (int mask = 1; mask < 32; mask <<= 1) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      if (r < Regs) {
        const float a = values[r];
        const float b = hadacore_shfl_xor_f32(a, mask);
        values[r] = (lane & mask) == 0 ? a + b : b - a;
      }
    }
  }
  if (Regs == 2) {
    const float a = values[0];
    const float b = values[1];
    values[0] = a + b;
    values[1] = a - b;
  }
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

template <int Width>
__device__ __forceinline__ void hadamard_fixed(float* values) {
#pragma unroll
  for (int stride = 1; stride < Width; stride <<= 1) {
    for (int pair = threadIdx.x; pair < Width / 2; pair += blockDim.x) {
      const int low = ((pair / stride) * (stride << 1)) +
                      (pair & (stride - 1));
      const float a = values[low];
      const float b = values[low + stride];
      values[low] = a + b;
      values[low + stride] = a - b;
    }
    __syncthreads();
  }
}

template <int Inner>
__device__ __forceinline__ void hadamard_grouped_fixed(
    float* values, int width) {
#pragma unroll
  for (int stride = 1; stride < Inner; stride <<= 1) {
    constexpr int pairs_per_group = Inner / 2;
    for (int pair = threadIdx.x; pair < width / 2; pair += blockDim.x) {
      const int group = pair / pairs_per_group;
      const int local_pair = pair - group * pairs_per_group;
      const int low = group * Inner +
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

template <int HeadDim>
__global__ void fused_rope_append_kv_i4_kernel(
    const half* query, const half* key, const half* value,
    const half* cos, const half* sin, half* query_out,
    uint8_t* kv_data, half2* kv_param,
    const int32_t* kv_indptr, const int32_t* kv_indices,
    const int32_t* last_page_offset, int num_layers, int layer_idx,
    int num_query_heads, int num_kv_heads, int page_size, int batch_size) {
  __shared__ float k_values[HeadDim];
  __shared__ float v_values[HeadDim];
  __shared__ float scratch[HeadDim];
  const int tid = threadIdx.x;
  const int batch = blockIdx.x;
  const int head = blockIdx.y;
  if (batch >= batch_size || head >= num_query_heads || tid >= HeadDim) return;

  const int half_dim = HeadDim / 2;
  const int partner = tid < half_dim ? tid + half_dim : tid - half_dim;
  const int trig = batch * HeadDim + tid;
  const int q_base = (batch * num_query_heads + head) * HeadDim;
  const half q = query[q_base + tid];
  const half qr = tid < half_dim ? __hneg(query[q_base + partner])
                                 : query[q_base + partner];
  query_out[q_base + tid] = __hadd(
      __hmul(q, cos[trig]), __hmul(qr, sin[trig]));

  // Only the first num_kv_heads blocks own K/V cache entries. This branch is
  // uniform across the block, so blocks for the remaining GQA query heads can
  // finish after writing rotated Q without entering any barrier.
  if (head >= num_kv_heads) return;

  const int seq_len =
      (kv_indptr[batch + 1] - kv_indptr[batch] - 1) * page_size +
      last_page_offset[batch];
  if (seq_len <= 0) return;
  const int page =
      kv_indices[kv_indptr[batch] + (seq_len - 1) / page_size];
  const int slot = (seq_len - 1) % page_size;
  const int kv_base = (batch * num_kv_heads + head) * HeadDim;
  const half k = key[kv_base + tid];
  const half kr = tid < half_dim ? __hneg(key[kv_base + partner])
                                 : key[kv_base + partner];
  // Preserve the old two-kernel numerical boundary: round RoPE output to FP16
  // before converting to FP32 for the Hadamard and cache quantization.
  const half rotated_k = __hadd(
      __hmul(k, cos[trig]), __hmul(kr, sin[trig]));
  k_values[tid] = __half2float(rotated_k);
  v_values[tid] = __half2float(value[kv_base + tid]);
  __syncthreads();

  hadamard(k_values, HeadDim);
  const float kx = k_values[tid] * rsqrtf(float(HeadDim));
  k_values[tid] = kx;
  const float kmax = reduce_max(kx, scratch);
  __syncthreads();
  const float kmin = -reduce_max(-kx, scratch);
  const half kscale_h =
      __float2half(fmaxf((kmax - kmin) / 15.0f, 1.0e-5f));
  const half kzero_h = __float2half(-kmin);
  const float kscale = __half2float(kscale_h);
  const float kzero = __half2float(kzero_h);
  if ((tid & 1) == 0) {
    const int q0 = max(0, min(15, static_cast<int>(
        nearbyintf((k_values[tid] + kzero) / kscale))));
    const int q1 = max(0, min(15, static_cast<int>(
        nearbyintf((k_values[tid + 1] + kzero) / kscale))));
    const size_t index =
        (((static_cast<size_t>(page) * num_layers + layer_idx) * 2 *
           num_kv_heads + head) * page_size + slot) *
        (HeadDim / 2) + (tid >> 1);
    kv_data[index] = pack_u4_pair(q0, q1);
  }
  if (tid == 0) {
    const size_t index =
        ((static_cast<size_t>(page) * num_layers + layer_idx) * 2 *
         num_kv_heads + head) * page_size + slot;
    kv_param[index] = __halves2half2(kscale_h, kzero_h);
  }
  __syncthreads();

  const float vx = v_values[tid];
  v_values[tid] = vx;
  const float vmax = reduce_max(vx, scratch);
  __syncthreads();
  const float vmin = -reduce_max(-vx, scratch);
  const half vscale_h =
      __float2half(fmaxf((vmax - vmin) / 15.0f, 1.0e-5f));
  const half vzero_h = __float2half(-vmin);
  const float vscale = __half2float(vscale_h);
  const float vzero = __half2float(vzero_h);
  if ((tid & 1) == 0) {
    const int q0 = max(0, min(15, static_cast<int>(
        nearbyintf((v_values[tid] + vzero) / vscale))));
    const int q1 = max(0, min(15, static_cast<int>(
        nearbyintf((v_values[tid + 1] + vzero) / vscale))));
    const size_t index =
        ((((static_cast<size_t>(page) * num_layers + layer_idx) * 2 + 1) *
           num_kv_heads + head) * page_size + slot) *
        (HeadDim / 2) + (tid >> 1);
    kv_data[index] = pack_u4_pair(q0, q1);
  }
  if (tid == 0) {
    const size_t index =
        (((static_cast<size_t>(page) * num_layers + layer_idx) * 2 + 1) *
         num_kv_heads + head) * page_size + slot;
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
  if (num_heads == 32 || num_heads == 64) {
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int waves = blockDim.x >> 5;
    const int regs = num_heads >> 5;
    for (int d = wave; d < head_dim; d += waves) {
      float registers[2] = {0.0f, 0.0f};
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if (r < regs) {
          const int h = r * 32 + lane;
          registers[r] = __half2float(
              attention[(row * num_heads + h) * head_dim + d]);
        }
      }
      if (regs == 1) {
        hadacore_warp_hadamard_regs<1>(registers, lane);
      } else {
        hadacore_warp_hadamard_regs<2>(registers, lane);
      }
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        if (r < regs) values[d * num_heads + r * 32 + lane] = registers[r];
      }
    }
    __syncthreads();
  } else {
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
      const int d = i / num_heads;
      const int h = i % num_heads;
      values[i] = __half2float(attention[(row * num_heads + h) * head_dim + d]);
    }
    __syncthreads();
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

__global__ void fused_attention_hadamard_quant_general_kernel(
    const half* attention, const half* matrix, uint8_t* packed, half* scales,
    int rows, int num_heads, int head_dim, int remainder, int inner) {
  extern __shared__ float values[];
  __shared__ float scratch[kThreads];
  const int row = blockIdx.x;
  const int width = num_heads * head_dim;
  if (row >= rows) return;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    const int d = i / num_heads;
    const int h = i - d * num_heads;
    values[i] = __half2float(attention[(row * num_heads + h) * head_dim + d]);
  }
  __syncthreads();
  for (int stride = 1; stride < inner; stride <<= 1) {
    const int pairs_per_group = inner >> 1;
    const int groups = head_dim * remainder;
    for (int pair = threadIdx.x; pair < groups * pairs_per_group;
         pair += blockDim.x) {
      const int group = pair / pairs_per_group;
      const int local = pair - group * pairs_per_group;
      const int low = group * inner + (local / stride) * (stride << 1)
                      + (local & (stride - 1));
      const float a = values[low], b = values[low + stride];
      values[low] = a + b;
      values[low + stride] = a - b;
    }
    __syncthreads();
  }
  constexpr int kOwned = (kMaxFusedWidth / 2 + kThreads - 1) / kThreads;
  float transformed[kOwned * 2];
  float local_max = 0.0f;
  const float norm = rsqrtf(float(num_heads));
  int owned = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned) {
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      const int flat = pair * 2 + which;
      const int h = flat / head_dim;
      const int d = flat - h * head_dim;
      const int out_k = h / inner;
      const int p = h - out_k * inner;
      float value = 0.0f;
      for (int in_k = 0; in_k < remainder; ++in_k) {
        value += __half2float(matrix[out_k * remainder + in_k])
                 * values[d * num_heads + in_k * inner + p];
      }
      value *= norm;
      transformed[owned * 2 + which] = value;
      local_max = fmaxf(local_max, fabsf(value));
    }
  }
  half scale_h = __float2half(fmaxf(
      reduce_max(local_max, scratch) / 7.0f, 6.103515625e-5f));
  const float scale = __half2float(scale_h);
  if (threadIdx.x == 0) scales[row] = scale_h;
  owned = 0;
  for (int pair = threadIdx.x; pair < width / 2;
       pair += blockDim.x, ++owned) {
    const int q0 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned * 2] / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(
        nearbyintf(transformed[owned * 2 + 1] / scale))));
    packed[row * (width / 2) + pair] = pack_s4_pair(q0, q1);
  }
}

// Experimental fusion-branch layout: each workgroup owns an independent
// 256-element block and emits one scale for that block. This is deliberately
// separate from the full-row QuaRot transform used by existing checkpoints.
template <int GroupSize>
__global__ void fused_ffn_silu_hadamard_quant_grouped_kernel(
    const half* gate, const half* up, uint8_t* packed, half* scales,
    int rows, int width) {
  __shared__ float values[GroupSize];
  __shared__ float scratch[GroupSize];
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int base = row * width + group * GroupSize;

  values[tid] = silu(__half2float(gate[base + tid])) *
                __half2float(up[base + tid]);
  __syncthreads();
  hadamard_fixed<GroupSize>(values);
  const float rotated = values[tid] * rsqrtf(static_cast<float>(GroupSize));
  values[tid] = rotated;
  const float scale = fmaxf(
      reduce_max(fabsf(rotated), scratch) / 7.0f, 6.103515625e-5f);
  const half scale_h = __float2half(scale);
  if (tid == 0) scales[row * gridDim.y + group] = scale_h;
  if ((tid & 1) == 0) {
    const int q0 = max(-8, min(7, static_cast<int>(
        nearbyintf(values[tid] / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(
        nearbyintf(values[tid + 1] / scale))));
    packed[(base + tid) >> 1] = pack_s4_pair(q0, q1);
  }
}

__global__ void fused_ffn_silu_hadamard_quant_hadacore256_kernel(
    const half* gate, const half* up, uint8_t* packed, half* scales,
    int rows, int width) {
  __shared__ uint16_t input[256];
  __shared__ uint16_t temporary[256];
  __shared__ uint16_t output[256];
  __shared__ float scratch[32];
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int base = row * width + group * 256;
  for (int i = lane; i < 256; i += 32)
    input[i] = quarot_hadamard_wmma::f32_to_bits(
        silu(__half2float(gate[base + i])) * __half2float(up[base + i]));
  __syncthreads();
  quarot_hadamard_wmma::h256(input, temporary, output, lane);
  float local_max = 0.0f;
  for (int i = lane; i < 256; i += 32)
    local_max = fmaxf(local_max, fabsf(quarot_hadamard_wmma::bits_to_f32(output[i])));
  scratch[lane] = local_max;
  __syncthreads();
  for (int stride = 16; stride > 0; stride >>= 1) {
    if (lane < stride) scratch[lane] = fmaxf(scratch[lane], scratch[lane + stride]);
    __syncthreads();
  }
  const float scale = fmaxf(scratch[0] / 7.0f, 6.103515625e-5f);
  const half scale_h = __float2half(scale);
  if (lane == 0) scales[row * gridDim.y + group] = scale_h;
  for (int pair = lane; pair < 128; pair += 32) {
    const float x0 = quarot_hadamard_wmma::bits_to_f32(output[pair * 2]);
    const float x1 = quarot_hadamard_wmma::bits_to_f32(output[pair * 2 + 1]);
    const int q0 = max(-8, min(7, static_cast<int>(nearbyintf(x0 / scale))));
    const int q1 = max(-8, min(7, static_cast<int>(nearbyintf(x1 / scale))));
    packed[(base + pair * 2) >> 1] = pack_s4_pair(q0, q1);
  }
}

// One wave owns H256. Each lane holds the eight elements whose indices share
// its low five bits. H32 uses wave shuffles; the remaining H8 stays entirely
// in registers. This is the same butterfly ordering as hadamard_fixed<256>.
// For width=K*P, QuaRot applies a power-of-two H_P followed by its
// fixed H_K remainder matrix. Recompute H_K for quantization to avoid a second
// width-sized LDS buffer (which exceeds 64 KiB for Llama-2 7B's width 11008).
// Split a large tensor-product transform at its natural boundary. Each block
// fuses SiLU*up with one inner power-of-two transform and uses only
// inner*sizeof(float) LDS. The finish kernel applies the remainder transform,
// computes the row scale, and directly packs INT4.
// gfx12 FP16 WMMA fragments used by the single-kernel remainder GEMM.
typedef _Float16 quarot_f16_frag __attribute__((ext_vector_type(8)));
typedef float quarot_f32_frag __attribute__((ext_vector_type(8)));

__device__ __forceinline__ quarot_f32_frag zero_quarot_f32_frag() {
  quarot_f32_frag result;
#pragma unroll
  for (int i = 0; i < 8; ++i) result[i] = 0.0f;
  return result;
}

__device__ __forceinline__ quarot_f32_frag quarot_wmma_f16(
    quarot_f16_frag a, quarot_f16_frag b, quarot_f32_frag c) {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1250__))
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
  return c;
#endif
}

__device__ __forceinline__ quarot_f16_frag load_remainder_a(
    const half* matrix, int remainder, int out_base, int in_base, int lane) {
  quarot_f16_frag frag;
  const int out = out_base + (lane & 15);
  const int k_base = in_base + (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int k = k_base + i;
    frag[i] = (out < remainder && k < remainder)
        ? static_cast<_Float16>(matrix[out * remainder + k])
        : static_cast<_Float16>(0.0f);
  }
  return frag;
}

__device__ __forceinline__ quarot_f16_frag load_remainder_b(
    const half* values, int remainder, int inner, int in_base, int p_base,
    int lane) {
  quarot_f16_frag frag;
  const int p = p_base + (lane & 15);
  const int k_base = in_base + (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int k = k_base + i;
    frag[i] = (k < remainder && p < inner)
        ? static_cast<_Float16>(values[k * inner + p])
        : static_cast<_Float16>(0.0f);
  }
  return frag;
}

// True single-kernel large FFN fusion for small generalized remainders.
// Storage stays FP16 in LDS; FHT and matrix accumulators use FP32 registers.
// One workgroup owns a row, so scale reduction and INT4 packing require no
// grid-wide synchronization or global intermediate.
void check_cuda_half_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA/HIP tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

bool power_of_two(int64_t x) { return x > 0 && (x & (x - 1)) == 0; }

}  // namespace

torch::Tensor fused_rope_append_kv_i4(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor cos, torch::Tensor sin, torch::Tensor kv_data,
    torch::Tensor kv_param, torch::Tensor kv_indptr, torch::Tensor kv_indices,
    torch::Tensor last_page_offset, int64_t num_layers, int64_t layer_idx,
    int64_t page_size) {
  check_cuda_half_contiguous(query, "query");
  check_cuda_half_contiguous(key, "key");
  check_cuda_half_contiguous(value, "value");
  check_cuda_half_contiguous(cos, "cos");
  check_cuda_half_contiguous(sin, "sin");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4 &&
                  query.size(1) == 1 && key.size(1) == 1 &&
                  value.sizes() == key.sizes() &&
                  query.size(0) == key.size(0) &&
                  query.size(3) == key.size(3),
              "K1 Q/K/V must be [batch, 1, heads, head_dim]");
  const int batch = query.size(0);
  const int query_heads = query.size(2);
  const int kv_heads = key.size(2);
  const int head_dim = query.size(3);
  TORCH_CHECK(query_heads % kv_heads == 0,
              "query heads must be divisible by KV heads");
  TORCH_CHECK((head_dim == 64 || head_dim == 128) &&
                  cos.numel() == batch * head_dim &&
                  sin.numel() == batch * head_dim,
              "invalid K1 RoPE shape");
  TORCH_CHECK(kv_data.is_cuda() && kv_data.is_contiguous() &&
                  kv_data.scalar_type() == torch::kUInt8,
              "kv_data must be contiguous uint8 CUDA/HIP");
  TORCH_CHECK(kv_param.is_cuda() && kv_param.is_contiguous() &&
                  kv_param.scalar_type() == torch::kFloat16,
              "kv_param must be contiguous float16 CUDA/HIP");
  TORCH_CHECK(kv_indptr.is_cuda() && kv_indices.is_cuda() &&
                  last_page_offset.is_cuda() &&
                  kv_indptr.is_contiguous() && kv_indices.is_contiguous() &&
                  last_page_offset.is_contiguous() &&
                  kv_indptr.scalar_type() == torch::kInt32 &&
                  kv_indices.scalar_type() == torch::kInt32 &&
                  last_page_offset.scalar_type() == torch::kInt32,
              "K1 paged-cache metadata must be contiguous int32 CUDA/HIP");
  TORCH_CHECK(kv_data.size(1) == num_layers && kv_data.size(3) == kv_heads &&
                  kv_data.size(4) == page_size &&
                  kv_data.size(5) * 2 == head_dim,
              "K1 cache dimensions do not match Q/K/V");
  TORCH_CHECK(kv_indptr.numel() == batch + 1 &&
                  last_page_offset.numel() == batch &&
                  layer_idx >= 0 && layer_idx < num_layers,
              "invalid K1 metadata or layer index");
  at::hip::HIPGuard guard{static_cast<char>(query.get_device())};
  auto query_out = torch::empty_like(query);
#define LAUNCH_FUSED_K1(HEAD_DIM) hipLaunchKernelGGL(\
      (fused_rope_append_kv_i4_kernel<HEAD_DIM>), \
      dim3(batch, query_heads), dim3(HEAD_DIM), 0, \
      at::hip::getCurrentHIPStream().stream(), \
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(key.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(value.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(cos.data_ptr<at::Half>()), \
      reinterpret_cast<const half*>(sin.data_ptr<at::Half>()), \
      reinterpret_cast<half*>(query_out.data_ptr<at::Half>()), \
      kv_data.data_ptr<uint8_t>(), \
      reinterpret_cast<half2*>(kv_param.data_ptr<at::Half>()), \
      kv_indptr.data_ptr<int32_t>(), kv_indices.data_ptr<int32_t>(), \
      last_page_offset.data_ptr<int32_t>(), num_layers, layer_idx, \
      query_heads, kv_heads, page_size, batch)
  if (head_dim == 64) { LAUNCH_FUSED_K1(64); }
  else { LAUNCH_FUSED_K1(128); }
#undef LAUNCH_FUSED_K1
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "fused_rope_append_kv_i4 launch failed");
  return query_out;
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

std::vector<torch::Tensor> fused_rmsnorm_quant_i4(
    torch::Tensor input, double eps, double clip_ratio) {
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
              "input must be a contiguous CUDA/HIP tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16,
              "input must be FP16");
  TORCH_CHECK(input.dim() >= 2 && input.size(-1) % 2 == 0,
              "last dimension must be even");
  const int cols = input.size(-1);
  const int rows = input.numel() / cols;
  auto packed_sizes = input.sizes().vec();
  packed_sizes.back() = cols / 2;
  auto scale_sizes = input.sizes().vec();
  scale_sizes.back() = 1;
  auto packed = torch::empty(
      packed_sizes, input.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_sizes, input.options());
  hipLaunchKernelGGL(fused_rmsnorm_quant_i4_kernel, dim3(rows),
                     dim3(kThreads), 0, at::hip::getCurrentHIPStream(),
                     reinterpret_cast<const half*>(input.data_ptr()),
                     packed.data_ptr<uint8_t>(),
                     reinterpret_cast<half*>(scales.data_ptr()), rows, cols,
                     static_cast<float>(eps), static_cast<float>(clip_ratio));
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "fused_rmsnorm_quant_i4 launch failed");
  return {packed, scales};
}

std::vector<torch::Tensor> fused_attention_hadamard_quant_general(
    torch::Tensor attention, int64_t num_heads, torch::Tensor matrix) {
  check_cuda_half_contiguous(attention, "attention");
  check_cuda_half_contiguous(matrix, "matrix");
  TORCH_CHECK(attention.dim() == 4 && attention.size(2) == num_heads,
              "attention must have shape [batch, sequence, heads, head_dim]");
  TORCH_CHECK(matrix.dim() == 2 && matrix.size(0) == matrix.size(1),
              "matrix must be square");
  TORCH_CHECK(attention.get_device() == matrix.get_device(),
              "attention and matrix must be on the same device");
  const int64_t remainder = matrix.size(0);
  TORCH_CHECK(num_heads % remainder == 0,
              "num_heads must be divisible by the remainder order");
  const int64_t inner = num_heads / remainder;
  TORCH_CHECK(power_of_two(inner), "heads / remainder must be a power of two");
  const int64_t head_dim = attention.size(3);
  const int64_t width = num_heads * head_dim;
  TORCH_CHECK(width <= kMaxFusedWidth && width % 2 == 0,
              "general fused attention supports even widths up to ",
              kMaxFusedWidth);
  const int64_t rows = attention.numel() / width;
  auto packed = torch::empty(
      {attention.size(0), attention.size(1), width / 2},
      attention.options().dtype(torch::kUInt8));
  auto scales = torch::empty(
      {attention.size(0), attention.size(1), 1}, attention.options());
  at::hip::HIPGuard guard{static_cast<char>(attention.get_device())};
  hipLaunchKernelGGL(fused_attention_hadamard_quant_general_kernel,
      dim3(rows), dim3(kThreads), width * sizeof(float),
      at::hip::getCurrentHIPStream().stream(),
      reinterpret_cast<const half*>(attention.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(matrix.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows,
      num_heads, head_dim, remainder, inner);
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "fused_attention_hadamard_quant_general launch failed");
  return {packed, scales};
}

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_grouped256(
    torch::Tensor gate, torch::Tensor up) {
  check_cuda_half_contiguous(gate, "gate");
  check_cuda_half_contiguous(up, "up");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.dim() >= 1,
              "gate and up must have identical non-empty shapes");
  const int64_t width = gate.size(-1);
  TORCH_CHECK(width > 0 && width % 256 == 0,
              "grouped H256 FFN requires a width divisible by 256");
  const int64_t rows = gate.numel() / width;
  auto packed_shape = gate.sizes().vec(); packed_shape.back() = width / 2;
  auto scale_shape = gate.sizes().vec(); scale_shape.back() = width / 256;
  auto packed = torch::empty(packed_shape,
                             gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, gate.options());
  at::hip::HIPGuard guard{static_cast<char>(gate.get_device())};
  auto stream = at::hip::getCurrentHIPStream().stream();
  if (rows > 1 || width >= 14336) {
    hipLaunchKernelGGL(fused_ffn_silu_hadamard_quant_hadacore256_kernel,
      dim3(rows, width / 256), dim3(32), 0, stream,
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, width);
  } else {
    hipLaunchKernelGGL((fused_ffn_silu_hadamard_quant_grouped_kernel<256>),
      dim3(rows, width / 256), dim3(256), 0, stream,
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, width);
  }
  TORCH_CHECK(hipGetLastError() == hipSuccess,
              "grouped H256 FFN launch failed");
  return {packed, scales};
}
