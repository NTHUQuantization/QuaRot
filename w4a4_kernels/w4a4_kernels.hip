#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kGroup = 128;
constexpr int kNBlock = 16;
constexpr int kFfn = 14336;

__device__ __forceinline__ int unpack_signed(uint8_t byte, bool high) {
  int value = high ? (byte >> 4) : (byte & 0xF);
  return value >= 8 ? value - 16 : value;
}

__device__ __forceinline__ uint8_t signed_nibble(int value) {
  return static_cast<uint8_t>(value) & 0xF;
}

__device__ __forceinline__ char4 unpack_signed4(uint16_t packed) {
  char4 value;
  int v0 = packed & 0xF;
  int v1 = (packed >> 4) & 0xF;
  int v2 = (packed >> 8) & 0xF;
  int v3 = (packed >> 12) & 0xF;
  value.x = static_cast<char>(v0 >= 8 ? v0 - 16 : v0);
  value.y = static_cast<char>(v1 >= 8 ? v1 - 16 : v1);
  value.z = static_cast<char>(v2 >= 8 ? v2 - 16 : v2);
  value.w = static_cast<char>(v3 >= 8 ? v3 - 16 : v3);
  return value;
}

__global__ void quantize_activation_kernel(
    const half* __restrict__ x,
    uint8_t* __restrict__ qdata,
    half* __restrict__ scales,
    int rows,
    int cols,
    int groups,
    float clip_ratio) {
  int group = blockIdx.x;
  int row = blockIdx.y;
  int tid = threadIdx.x;
  if (row >= rows || group >= groups || tid >= kGroup) return;

  __shared__ float reduce[kGroup];
  __shared__ float shared_scale;
  int base = row * cols + group * kGroup;
  float value = __half2float(x[base + tid]);
  reduce[tid] = fabsf(value);
  __syncthreads();
  for (int stride = kGroup / 2; stride; stride >>= 1) {
    if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    __syncthreads();
  }
  if (tid == 0) {
    float amax = reduce[0];
    shared_scale = amax == 0.f ? 1.f : fmaxf(amax * clip_ratio / 7.f, 1.0e-6f);
    scales[row * groups + group] = __float2half(shared_scale);
  }
  __syncthreads();

  if ((tid & 1) == 0) {
    float other = __half2float(x[base + tid + 1]);
    int q0 = max(-8, min(7, static_cast<int>(nearbyintf(value / shared_scale))));
    int q1 = max(-8, min(7, static_cast<int>(nearbyintf(other / shared_scale))));
    qdata[row * (cols / 2) + group * (kGroup / 2) + tid / 2] =
        signed_nibble(q0) | (signed_nibble(q1) << 4);
  }
}

__global__ void rmsnorm_quantize_kernel(
    const half* __restrict__ x,
    uint8_t* __restrict__ qdata,
    half* __restrict__ scales,
    int rows,
    int cols,
    int groups,
    float epsilon,
    float clip_ratio) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows || tid >= kGroup) return;
  __shared__ float reduce[kGroup];
  __shared__ float inv_rms;
  __shared__ float group_scale;
  float sum = 0.f;
  for (int k = tid; k < cols; k += kGroup) {
    float value = __half2float(x[row * cols + k]);
    sum += value * value;
  }
  reduce[tid] = sum;
  __syncthreads();
  for (int stride = kGroup / 2; stride; stride >>= 1) {
    if (tid < stride) reduce[tid] += reduce[tid + stride];
    __syncthreads();
  }
  if (tid == 0) inv_rms = rsqrtf(reduce[0] / static_cast<float>(cols) + epsilon);
  __syncthreads();
  for (int group = 0; group < groups; ++group) {
    int k = group * kGroup + tid;
    float value = __half2float(x[row * cols + k]) * inv_rms;
    reduce[tid] = fabsf(value);
    __syncthreads();
    for (int stride = kGroup / 2; stride; stride >>= 1) {
      if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
      __syncthreads();
    }
    if (tid == 0) {
      float amax = reduce[0];
      group_scale = amax == 0.f ? 1.f : fmaxf(amax * clip_ratio / 7.f, 1.0e-6f);
      scales[row * groups + group] = __float2half(group_scale);
    }
    __syncthreads();
    if ((tid & 1) == 0) {
      float other = __half2float(x[row * cols + k + 1]) * inv_rms;
      int q0 = max(-8, min(7, static_cast<int>(nearbyintf(value / group_scale))));
      int q1 = max(-8, min(7, static_cast<int>(nearbyintf(other / group_scale))));
      qdata[row * (cols / 2) + group * (kGroup / 2) + tid / 2] =
          signed_nibble(q0) | (signed_nibble(q1) << 4);
    }
    __syncthreads();
  }
}

__global__ void silu_hadamard_quantize_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    const float* __restrict__ h28,
    uint8_t* __restrict__ qdata,
    half* __restrict__ scales,
    int rows,
    float clip_ratio) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows || tid >= 512) return;
  __shared__ float data[kFfn];
  __shared__ float reduce[kGroup];
  int row_base = row * kFfn;
  for (int index = tid; index < kFfn; index += 512) {
    float g = __half2float(gate[row_base + index]);
    float u = __half2float(up[row_base + index]);
    data[index] = (g / (1.f + expf(-g))) * u;
  }
  __syncthreads();

  // H512 on each of the 28 independent segments.
  for (int stride = 1; stride < 512; stride <<= 1) {
    for (int pair = tid; pair < kFfn / 2; pair += 512) {
      int segment = pair / 256;
      int within = pair - segment * 256;
      int block = (within / stride) * (stride * 2);
      int offset = within % stride;
      int first = segment * 512 + block + offset;
      int second = first + stride;
      float a = data[first];
      float b = data[second];
      data[first] = a + b;
      data[second] = a - b;
    }
    __syncthreads();
  }

  // H28 across segments. Every thread owns one column and loads all source
  // values before writing, so the in-place transform has no cross-thread race.
  constexpr float kInvSqrt512 = 0.04419417382415922f;
  if (tid < 512) {
    float source[28];
    #pragma unroll
    for (int segment = 0; segment < 28; ++segment) {
      source[segment] = data[segment * 512 + tid] * kInvSqrt512;
    }
    #pragma unroll
    for (int output_segment = 0; output_segment < 28; ++output_segment) {
      float sum = source[0] * h28[output_segment * 28];
      #pragma unroll
      for (int input_segment = 1; input_segment < 28; ++input_segment) {
        sum += source[input_segment] * h28[output_segment * 28 + input_segment];
      }
      data[output_segment * 512 + tid] = sum;
    }
  }
  __syncthreads();

  constexpr int kGroups = kFfn / kGroup;
  for (int group = 0; group < kGroups; ++group) {
    if (tid < kGroup) reduce[tid] = fabsf(data[group * kGroup + tid]);
    __syncthreads();
    for (int stride = kGroup / 2; stride; stride >>= 1) {
      if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
      __syncthreads();
    }
    float scale = reduce[0] == 0.f ? 1.f : fmaxf(reduce[0] * clip_ratio / 7.f, 1.0e-6f);
    if (tid == 0) scales[row * kGroups + group] = __float2half(scale);
    if (tid < kGroup / 2) {
      int first = group * kGroup + tid * 2;
      int q0 = max(-8, min(7, static_cast<int>(nearbyintf(data[first] / scale))));
      int q1 = max(-8, min(7, static_cast<int>(nearbyintf(data[first + 1] / scale))));
      qdata[row * (kFfn / 2) + group * (kGroup / 2) + tid] =
          signed_nibble(q0) | (signed_nibble(q1) << 4);
    }
    __syncthreads();
  }
}

__global__ void qkv_rope_hadamard_kernel(
    const half* __restrict__ qkv,
    const int64_t* __restrict__ positions,
    half* __restrict__ q,
    half* __restrict__ k,
    half* __restrict__ v,
    int batch,
    int tokens,
    float theta) {
  int vector = blockIdx.x;
  int tid = threadIdx.x;
  constexpr int kVectors = 48;
  if (tid >= 128 || vector >= batch * tokens * kVectors) return;
  int batch_index = vector / (tokens * kVectors);
  int remainder = vector - batch_index * tokens * kVectors;
  int token = remainder / kVectors;
  int head_kind = remainder - token * kVectors;
  int source_offset;
  half* destination;
  int destination_heads;
  int destination_head;
  if (head_kind < 32) {
    source_offset = head_kind * 128;
    destination = q;
    destination_heads = 32;
    destination_head = head_kind;
  } else if (head_kind < 40) {
    source_offset = 4096 + (head_kind - 32) * 128;
    destination = k;
    destination_heads = 8;
    destination_head = head_kind - 32;
  } else {
    source_offset = 5120 + (head_kind - 40) * 128;
    destination = v;
    destination_heads = 8;
    destination_head = head_kind - 40;
  }
  int source_base = (batch_index * tokens + token) * 6144 + source_offset;
  size_t destination_index =
      ((static_cast<size_t>(batch_index) * destination_heads + destination_head) * tokens + token) * 128 + tid;
  if (head_kind >= 40) {
    destination[destination_index] = qkv[source_base + tid];
    return;
  }

  __shared__ float data[128];
  int pair = tid & 63;
  float first = __half2float(qkv[source_base + pair]);
  float second = __half2float(qkv[source_base + pair + 64]);
  float angle = static_cast<float>(positions[token]) * powf(theta, -static_cast<float>(pair) / 64.f);
  float sine, cosine;
  sincosf(angle, &sine, &cosine);
  data[tid] = tid < 64 ? first * cosine - second * sine : second * cosine + first * sine;
  __syncthreads();
  for (int stride = 1; stride < 128; stride <<= 1) {
    int block = (tid / (stride * 2)) * (stride * 2);
    int offset = tid % stride;
    if ((tid % (stride * 2)) < stride) {
      int lhs = block + offset;
      int rhs = lhs + stride;
      float a = data[lhs];
      float b = data[rhs];
      data[lhs] = a + b;
      data[rhs] = a - b;
    }
    __syncthreads();
  }
  destination[destination_index] = __float2half(data[tid] * 0.08838834764831845f);
}

__global__ void cross_head_hadamard_kernel(
    const half* __restrict__ attention,
    half* __restrict__ output,
    int vectors) {
  int vector = blockIdx.x;
  int tid = threadIdx.x;
  if (vector >= vectors || tid >= 32) return;
  __shared__ float data[32];
  int token_vector = vector / 128;
  int dimension = vector - token_vector * 128;
  size_t base = static_cast<size_t>(token_vector) * 32 * 128 + dimension;
  data[tid] = __half2float(attention[base + tid * 128]);
  __syncthreads();
  for (int stride = 1; stride < 32; stride <<= 1) {
    if ((tid % (stride * 2)) < stride) {
      int rhs = tid + stride;
      float a = data[tid];
      float b = data[rhs];
      data[tid] = a + b;
      data[rhs] = a - b;
    }
    __syncthreads();
  }
  output[base + tid * 128] = __float2half(data[tid] * 0.1767766952966369f);
}

__global__ void quantize_kv_biased_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    uint8_t* __restrict__ packed_key,
    half* __restrict__ key_params,
    uint8_t* __restrict__ packed_value,
    half* __restrict__ value_params,
    int vectors,
    float clip_ratio) {
  int vector = blockIdx.x;
  int tid = threadIdx.x;
  if (vector >= vectors || tid >= 128) return;
  __shared__ float reduce[128];
  __shared__ float scale;
  const half* sources[2] = {key, value};
  uint8_t* packed[2] = {packed_key, packed_value};
  half* params[2] = {key_params, value_params};
  for (int kind = 0; kind < 2; ++kind) {
    float item = __half2float(sources[kind][static_cast<size_t>(vector) * 128 + tid]);
    reduce[tid] = fabsf(item);
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
      if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
      __syncthreads();
    }
    if (tid == 0) {
      float amax = reduce[0];
      scale = amax == 0.f ? 1.f : fmaxf(amax * clip_ratio / 7.f, 1.0e-6f);
      params[kind][vector * 2] = __float2half(scale);
      params[kind][vector * 2 + 1] = __float2half(scale * 8.f);
    }
    __syncthreads();
    if (tid < 64) {
      float first = __half2float(sources[kind][static_cast<size_t>(vector) * 128 + tid * 2]);
      float second = __half2float(sources[kind][static_cast<size_t>(vector) * 128 + tid * 2 + 1]);
      int q0 = max(-8, min(7, static_cast<int>(nearbyintf(first / scale))));
      int q1 = max(-8, min(7, static_cast<int>(nearbyintf(second / scale))));
      packed[kind][static_cast<size_t>(vector) * 64 + tid] =
          static_cast<uint8_t>(q0 + 8) | (static_cast<uint8_t>(q1 + 8) << 4);
    }
    __syncthreads();
  }
}

__global__ void append_kv_biased_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    uint8_t* __restrict__ cache_data,
    half* __restrict__ cache_params,
    const int32_t* __restrict__ page_ids,
    int batch,
    int page_size,
    int slot,
    float clip_ratio) {
  int vector = blockIdx.x;
  int tid = threadIdx.x;
  constexpr int kHeads = 8;
  if (vector >= batch * kHeads || tid >= 128) return;
  int batch_index = vector / kHeads;
  int head = vector - batch_index * kHeads;
  int page = page_ids[batch_index];
  __shared__ float reduce[128];
  __shared__ float scale;
  const half* sources[2] = {key, value};
  for (int kind = 0; kind < 2; ++kind) {
    float item = __half2float(sources[kind][static_cast<size_t>(vector) * 128 + tid]);
    reduce[tid] = fabsf(item);
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
      if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
      __syncthreads();
    }
    size_t params_base =
        ((((static_cast<size_t>(page) * 2 + kind) * kHeads + head) * page_size + slot) * 2);
    if (tid == 0) {
      float amax = reduce[0];
      scale = amax == 0.f ? 1.f : fmaxf(amax * clip_ratio / 7.f, 1.0e-6f);
      cache_params[params_base] = __float2half(scale);
      cache_params[params_base + 1] = __float2half(scale * 8.f);
    }
    __syncthreads();
    if (tid < 64) {
      float first = __half2float(sources[kind][static_cast<size_t>(vector) * 128 + tid * 2]);
      float second = __half2float(sources[kind][static_cast<size_t>(vector) * 128 + tid * 2 + 1]);
      int q0 = max(-8, min(7, static_cast<int>(nearbyintf(first / scale))));
      int q1 = max(-8, min(7, static_cast<int>(nearbyintf(second / scale))));
      size_t data_base =
          ((((static_cast<size_t>(page) * 2 + kind) * kHeads + head) * page_size + slot) * 64);
      cache_data[data_base + tid] =
          static_cast<uint8_t>(q0 + 8) | (static_cast<uint8_t>(q1 + 8) << 4);
    }
    __syncthreads();
  }
}

__global__ void append_kv_chunk_biased_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    uint8_t* __restrict__ cache_data,
    half* __restrict__ cache_params,
    const int32_t* __restrict__ allocation_indptr,
    const int32_t* __restrict__ allocation_indices,
    int batch,
    int chunk,
    int page_size,
    int start_position,
    float clip_ratio) {
  int vector = blockIdx.x;
  int tid = threadIdx.x;
  constexpr int kHeads = 8;
  if (vector >= batch * chunk * kHeads || tid >= 128) return;
  int head = vector % kHeads;
  int request_token = vector / kHeads;
  int offset = request_token % chunk;
  int batch_index = request_token / chunk;
  int position = start_position + offset;
  int page = allocation_indices[allocation_indptr[batch_index] + position / page_size];
  int slot = position % page_size;
  __shared__ float reduce[128];
  __shared__ float scale;
  const half* sources[2] = {key, value};
  for (int kind = 0; kind < 2; ++kind) {
    float item = __half2float(sources[kind][static_cast<size_t>(vector) * 128 + tid]);
    reduce[tid] = fabsf(item);
    __syncthreads();
    for (int stride = 64; stride; stride >>= 1) {
      if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
      __syncthreads();
    }
    size_t params_base =
        ((((static_cast<size_t>(page) * 2 + kind) * kHeads + head) * page_size + slot) * 2);
    if (tid == 0) {
      float amax = reduce[0];
      scale = amax == 0.f ? 1.f : fmaxf(amax * clip_ratio / 7.f, 1.0e-6f);
      cache_params[params_base] = __float2half(scale);
      cache_params[params_base + 1] = __float2half(scale * 8.f);
    }
    __syncthreads();
    if (tid < 64) {
      float first = __half2float(sources[kind][static_cast<size_t>(vector) * 128 + tid * 2]);
      float second = __half2float(sources[kind][static_cast<size_t>(vector) * 128 + tid * 2 + 1]);
      int q0 = max(-8, min(7, static_cast<int>(nearbyintf(first / scale))));
      int q1 = max(-8, min(7, static_cast<int>(nearbyintf(second / scale))));
      size_t data_base =
          ((((static_cast<size_t>(page) * 2 + kind) * kHeads + head) * page_size + slot) * 64);
      cache_data[data_base + tid] =
          static_cast<uint8_t>(q0 + 8) | (static_cast<uint8_t>(q1 + 8) << 4);
    }
    __syncthreads();
  }
}

__global__ void virtual_verify_metadata_kernel(
    int32_t* __restrict__ indptr,
    int32_t* __restrict__ indices,
    int32_t* __restrict__ last_page_offset,
    int batch,
    int start_len,
    int chunk,
    int page_size,
    int pages_per_batch) {
  if (blockIdx.x || threadIdx.x) return;
  int request_count = batch * chunk;
  int cursor = 0;
  indptr[0] = 0;
  for (int request = 0; request < request_count; ++request) {
    int batch_index = request / chunk;
    int offset = request - batch_index * chunk;
    int length = start_len + offset + 1;
    int used_pages = (length + page_size - 1) / page_size;
    int first_page = batch_index * pages_per_batch;
    for (int page = 0; page < used_pages; ++page) indices[cursor++] = first_page + page;
    indptr[request + 1] = cursor;
    last_page_offset[request] = length - (used_pages - 1) * page_size;
  }
}

// A wave32 computes a small output tile. Each lane handles four K values
// through RDNA's packed signed INT8 dot instruction. N=4/8/16 variants allow
// gfx1201 to trade activation rereads for occupancy / VGPR pressure.
template <int Outputs>
__global__ void w4a4_linear_dot4_kernel(
    const uint8_t* __restrict__ qx,
    const half* __restrict__ sx,
    const uint8_t* __restrict__ qw,
    const half* __restrict__ sw,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int rows,
    int out_features,
    int in_features,
    int groups,
    bool has_bias) {
  int output_base = blockIdx.x * Outputs;
  int row = blockIdx.y;
  int tid = threadIdx.x;
  if (output_base >= out_features || row >= rows || tid >= 32) return;

  float result[Outputs];
  if (tid == 0) {
    #pragma unroll
    for (int output = 0; output < Outputs; ++output) {
      int n = output_base + output;
      result[output] = has_bias ? __half2float(bias[n]) : 0.f;
    }
  }
  for (int group = 0; group < groups; ++group) {
    const uint8_t* xptr = qx + row * (in_features / 2) + group * 64 + tid * 2;
    uint16_t xword = static_cast<uint16_t>(xptr[0]) | (static_cast<uint16_t>(xptr[1]) << 8);
    char4 x4 = unpack_signed4(xword);
    float xscale = __half2float(sx[row * groups + group]);
    #pragma unroll
    for (int output = 0; output < Outputs; ++output) {
      int n = output_base + output;
      int ntile = n / kNBlock;
      int ninner = n % kNBlock;
      size_t woffset = (((static_cast<size_t>(ntile) * groups + group) * kNBlock + ninner) * 64) + tid * 2;
      uint16_t wword = static_cast<uint16_t>(qw[woffset]) |
                       (static_cast<uint16_t>(qw[woffset + 1]) << 8);
      int sum = amd_mixed_dot(x4, unpack_signed4(wword), 0, false);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
      }
      if (tid == 0) {
        result[output] += static_cast<float>(sum) * xscale *
                          __half2float(sw[n * groups + group]);
      }
    }
  }
  if (tid == 0) {
    #pragma unroll
    for (int item = 0; item < Outputs; ++item) {
      output[row * out_features + output_base + item] = __float2half(result[item]);
    }
  }
}

// Four independent waves per workgroup retain the low-VGPR N=2 computation
// while presenting contiguous eight-output blocks to gfx1201's schedulers.
__global__ void w4a4_linear_dot4_n2_waves4_kernel(
    const uint8_t* __restrict__ qx,
    const half* __restrict__ sx,
    const uint8_t* __restrict__ qw,
    const half* __restrict__ sw,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int out_features,
    int in_features,
    int groups,
    bool has_bias) {
  constexpr int kOutputs = 2;
  int wave = threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  int output_base = (blockIdx.x * 4 + wave) * kOutputs;
  float result[kOutputs];
  if (lane == 0) {
    result[0] = has_bias ? __half2float(bias[output_base]) : 0.f;
    result[1] = has_bias ? __half2float(bias[output_base + 1]) : 0.f;
  }
  for (int group = 0; group < groups; ++group) {
    const uint8_t* xptr = qx + group * 64 + lane * 2;
    uint16_t xword = static_cast<uint16_t>(xptr[0]) | (static_cast<uint16_t>(xptr[1]) << 8);
    char4 x4 = unpack_signed4(xword);
    float xscale = __half2float(sx[group]);
    #pragma unroll
    for (int item = 0; item < kOutputs; ++item) {
      int n = output_base + item;
      int ntile = n / kNBlock;
      int ninner = n % kNBlock;
      size_t woffset = (((static_cast<size_t>(ntile) * groups + group) * kNBlock + ninner) * 64) +
                       lane * 2;
      uint16_t wword = static_cast<uint16_t>(qw[woffset]) |
                       (static_cast<uint16_t>(qw[woffset + 1]) << 8);
      int sum = amd_mixed_dot(x4, unpack_signed4(wword), 0, false);
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
      }
      if (lane == 0) {
        result[item] += static_cast<float>(sum) * xscale * __half2float(sw[n * groups + group]);
      }
    }
  }
  if (lane == 0) {
    output[output_base] = __float2half(result[0]);
    output[output_base + 1] = __float2half(result[1]);
  }
}

// M>1 path: four waves compute four activation rows while sharing one W4
// output tile through LDS. This avoids rereading the same weights once per row
// in speculative verification and prefill.
template <int RowsPerBlock>
__global__ void w4a4_linear_dot4_rows_kernel(
    const uint8_t* __restrict__ qx,
    const half* __restrict__ sx,
    const uint8_t* __restrict__ qw,
    const half* __restrict__ sw,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int rows,
    int out_features,
    int in_features,
    int groups,
    bool has_bias) {
  constexpr int kOutputs = 4;
  static_assert(RowsPerBlock == 4 || RowsPerBlock == 8, "unsupported row tile");
  int output_base = blockIdx.x * kOutputs;
  int wave = threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  int row = blockIdx.y * RowsPerBlock + wave;
  bool valid_row = row < rows;
  __shared__ uint16_t shared_weight[kOutputs][32];
  float result[kOutputs];
  #pragma unroll
  for (int item = 0; item < kOutputs; ++item) {
    int n = output_base + item;
    result[item] = (valid_row && lane == 0 && has_bias) ? __half2float(bias[n]) : 0.f;
  }

  for (int group = 0; group < groups; ++group) {
    if (threadIdx.x < kOutputs * 32) {
      int load_output = threadIdx.x / 32;
      int load_lane = threadIdx.x & 31;
      int n = output_base + load_output;
      int ntile = n / kNBlock;
      int ninner = n % kNBlock;
      size_t woffset = (((static_cast<size_t>(ntile) * groups + group) * kNBlock + ninner) * 64) +
                       load_lane * 2;
      shared_weight[load_output][load_lane] = static_cast<uint16_t>(qw[woffset]) |
          (static_cast<uint16_t>(qw[woffset + 1]) << 8);
    }
    __syncthreads();

    char4 x4{};
    float xscale = 0.f;
    if (valid_row) {
      const uint8_t* xptr = qx + row * (in_features / 2) + group * 64 + lane * 2;
      uint16_t xword = static_cast<uint16_t>(xptr[0]) | (static_cast<uint16_t>(xptr[1]) << 8);
      x4 = unpack_signed4(xword);
      xscale = __half2float(sx[row * groups + group]);
    }
    #pragma unroll
    for (int item = 0; item < kOutputs; ++item) {
      int sum = valid_row ? amd_mixed_dot(x4, unpack_signed4(shared_weight[item][lane]), 0, false) : 0;
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
      }
      if (valid_row && lane == 0) {
        result[item] += static_cast<float>(sum) * xscale *
                        __half2float(sw[(output_base + item) * groups + group]);
      }
    }
    __syncthreads();
  }
  if (valid_row && lane == 0) {
    #pragma unroll
    for (int item = 0; item < kOutputs; ++item) {
      output[row * out_features + output_base + item] = __float2half(result[item]);
    }
  }
}

// gfx1201 speculative verification path. rocWMMA operates on 16x16x16 I8
// tiles, so M=2..4, 5..8 and 9..16 dispatch to separately compiled padded
// variants. One wave owns an N=16 output tile. Packed W4 is expanded only into
// LDS, activation / weight tiles are shared by all 256 output accumulators,
// and each K=128 quantization group is scaled independently after INT32 MMA.
template <int PadM>
__global__ void w4a4_linear_wmma_i8_kernel(
    const uint8_t* __restrict__ qx,
    const half* __restrict__ sx,
    const uint8_t* __restrict__ qw,
    const half* __restrict__ sw,
    const half* __restrict__ bias,
    half* __restrict__ output,
    int rows,
    int out_features,
    int in_features,
    int groups,
    bool has_bias) {
  static_assert(PadM == 4 || PadM == 8 || PadM == 16, "unsupported verification bucket");
  using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, int8_t, rocwmma::row_major>;
  using FragB = rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, int8_t, rocwmma::col_major>;
  using FragC = rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, int32_t>;
  int lane = threadIdx.x;
  int ntile = blockIdx.x;
  __shared__ int8_t tile_a[16 * 128];
  // Stored as [N, K], which is column-major storage for mathematical [K, N].
  __shared__ int8_t tile_b[16 * 128];
  __shared__ int32_t tile_c[16 * 16];
  float accumulated[8];
  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    int linear = lane + item * 32;
    int row = linear / 16;
    int n = ntile * 16 + linear % 16;
    accumulated[item] = (row < rows && row < PadM && has_bias) ? __half2float(bias[n]) : 0.f;
  }

  for (int group = 0; group < groups; ++group) {
    for (int linear = lane; linear < 16 * 128; linear += 32) {
      int row = linear / 128;
      int k = linear - row * 128;
      if (row < rows && row < PadM) {
        uint8_t packed = qx[row * (in_features / 2) + group * 64 + k / 2];
        tile_a[linear] = static_cast<int8_t>(unpack_signed(packed, k & 1));
      } else {
        tile_a[linear] = 0;
      }
      int ninner = row;
      size_t woffset = (((static_cast<size_t>(ntile) * groups + group) * 16 + ninner) * 64) + k / 2;
      uint8_t packed_weight = qw[woffset];
      tile_b[linear] = static_cast<int8_t>(unpack_signed(packed_weight, k & 1));
    }
    __syncthreads();

    FragC fragment_c;
    rocwmma::fill_fragment(fragment_c, 0);
    #pragma unroll
    for (int kbase = 0; kbase < 128; kbase += 16) {
      FragA fragment_a;
      FragB fragment_b;
      rocwmma::load_matrix_sync(fragment_a, tile_a + kbase, 128);
      rocwmma::load_matrix_sync(fragment_b, tile_b + kbase, 128);
      rocwmma::mma_sync(fragment_c, fragment_a, fragment_b, fragment_c);
    }
    rocwmma::store_matrix_sync(tile_c, fragment_c, 16, rocwmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      int linear = lane + item * 32;
      int row = linear / 16;
      int ninner = linear % 16;
      if (row < rows && row < PadM) {
        accumulated[item] += static_cast<float>(tile_c[linear]) *
            __half2float(sx[row * groups + group]) *
            __half2float(sw[(ntile * 16 + ninner) * groups + group]);
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int item = 0; item < 8; ++item) {
    int linear = lane + item * 32;
    int row = linear / 16;
    int n = ntile * 16 + linear % 16;
    if (row < rows && row < PadM) output[row * out_features + n] = __float2half(accumulated[item]);
  }
}

__global__ void embedding_lookup_kernel(
    const int64_t* __restrict__ ids,
    const uint8_t* __restrict__ qw,
    const half* __restrict__ scales,
    half* __restrict__ output,
    int tokens,
    int vocab,
    int cols,
    int groups) {
  int token = blockIdx.x;
  int group = blockIdx.y;
  int tid = threadIdx.x;
  if (token >= tokens || group >= groups || tid >= kGroup) return;
  int64_t row = ids[token];
  if (row < 0 || row >= vocab) return;
  int k = group * kGroup + tid;
  uint8_t byte = qw[row * (cols / 2) + k / 2];
  int q = unpack_signed(byte, tid & 1);
  output[token * cols + k] = __float2half(static_cast<float>(q) * __half2float(scales[row * groups + group]));
}

void check_half_cuda(const torch::Tensor& value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be on a HIP device");
  TORCH_CHECK(value.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> quantize_activation(torch::Tensor x, int64_t group_size, double clip_ratio) {
  check_half_cuda(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be [rows, cols]");
  TORCH_CHECK(group_size == kGroup, "only group_size=128 is supported");
  TORCH_CHECK(x.size(1) % kGroup == 0, "cols must be divisible by 128");
  TORCH_CHECK(clip_ratio > 0.0 && clip_ratio <= 1.0, "clip_ratio must be in (0, 1]");
  int rows = x.size(0), cols = x.size(1), groups = cols / kGroup;
  auto qdata = torch::empty({rows, cols / 2}, x.options().dtype(torch::kUInt8));
  auto scales = torch::empty({rows, groups}, x.options());
  hipLaunchKernelGGL(quantize_activation_kernel, dim3(groups, rows), dim3(kGroup), 0, 0,
                     reinterpret_cast<const half*>(x.data_ptr<at::Half>()), qdata.data_ptr<uint8_t>(),
                     reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, cols, groups,
                     static_cast<float>(clip_ratio));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {qdata, scales};
}

std::vector<torch::Tensor> rmsnorm_quantize(
    torch::Tensor x, double epsilon, int64_t group_size, double clip_ratio) {
  check_half_cuda(x, "x");
  TORCH_CHECK(x.dim() == 2, "x must be [rows, cols]");
  TORCH_CHECK(group_size == kGroup, "only group_size=128 is supported");
  TORCH_CHECK(x.size(1) % kGroup == 0, "cols must be divisible by 128");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  int rows = x.size(0), cols = x.size(1), groups = cols / kGroup;
  auto qdata = torch::empty({rows, cols / 2}, x.options().dtype(torch::kUInt8));
  auto scales = torch::empty({rows, groups}, x.options());
  hipLaunchKernelGGL(rmsnorm_quantize_kernel, dim3(rows), dim3(kGroup), 0, 0,
                     reinterpret_cast<const half*>(x.data_ptr<at::Half>()), qdata.data_ptr<uint8_t>(),
                     reinterpret_cast<half*>(scales.data_ptr<at::Half>()), rows, cols, groups,
                     static_cast<float>(epsilon), static_cast<float>(clip_ratio));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {qdata, scales};
}

std::vector<torch::Tensor> silu_hadamard_quantize(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor h28,
    int64_t group_size,
    double clip_ratio) {
  check_half_cuda(gate, "gate");
  check_half_cuda(up, "up");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.dim() == 2 && gate.size(1) == kFfn,
              "gate/up must be [rows, 14336]");
  TORCH_CHECK(h28.is_cuda() && h28.scalar_type() == torch::kFloat32 && h28.is_contiguous() &&
              h28.sizes() == torch::IntArrayRef({28, 28}), "h28 must be HIP float32 [28, 28]");
  TORCH_CHECK(group_size == kGroup, "only group_size=128 is supported");
  int rows = gate.size(0);
  auto qdata = torch::empty({rows, kFfn / 2}, gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty({rows, kFfn / kGroup}, gate.options());
  hipLaunchKernelGGL(silu_hadamard_quantize_kernel, dim3(rows), dim3(512), 0, 0,
                     reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
                     reinterpret_cast<const half*>(up.data_ptr<at::Half>()), h28.data_ptr<float>(),
                     qdata.data_ptr<uint8_t>(), reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
                     rows, static_cast<float>(clip_ratio));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {qdata, scales};
}

torch::Tensor w4a4_linear(
    torch::Tensor qx,
    torch::Tensor sx,
    torch::Tensor qw,
    torch::Tensor sw,
    torch::Tensor bias,
    int64_t output_tile,
    int64_t decode_waves,
    int64_t parallel_kernel) {
  TORCH_CHECK(qx.is_cuda() && qw.is_cuda(), "packed operands must be on a HIP device");
  TORCH_CHECK(qx.scalar_type() == torch::kUInt8 && qw.scalar_type() == torch::kUInt8,
              "packed operands must be uint8");
  check_half_cuda(sx, "activation scales");
  check_half_cuda(sw, "weight scales");
  TORCH_CHECK(qx.dim() == 2 && qw.dim() == 4 && sx.dim() == 2 && sw.dim() == 2,
              "invalid W4A4 tensor ranks");
  int rows = qx.size(0), in_features = qx.size(1) * 2;
  int out_features = qw.size(0) * kNBlock, groups = in_features / kGroup;
  TORCH_CHECK(qw.size(1) == groups && qw.size(2) == kNBlock && qw.size(3) == kGroup / 2,
              "invalid blocked weight layout");
  TORCH_CHECK(sx.sizes() == torch::IntArrayRef({rows, groups}), "invalid activation scales");
  TORCH_CHECK(sw.sizes() == torch::IntArrayRef({out_features, groups}), "invalid weight scales");
  bool has_bias = bias.numel() != 0;
  if (has_bias) {
    check_half_cuda(bias, "bias");
    TORCH_CHECK(bias.numel() == out_features, "invalid bias shape");
  }
  auto output = torch::empty({rows, out_features}, sx.options());
  const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;
  if (rows > 1 && rows <= 16 && parallel_kernel == 1) {
    if (rows <= 4) {
      hipLaunchKernelGGL((w4a4_linear_wmma_i8_kernel<4>), dim3(out_features / 16), dim3(32), 0, 0,
                         qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                         qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                         bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                         in_features, groups, has_bias);
    } else if (rows <= 8) {
      hipLaunchKernelGGL((w4a4_linear_wmma_i8_kernel<8>), dim3(out_features / 16), dim3(32), 0, 0,
                         qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                         qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                         bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                         in_features, groups, has_bias);
    } else {
      hipLaunchKernelGGL((w4a4_linear_wmma_i8_kernel<16>), dim3(out_features / 16), dim3(32), 0, 0,
                         qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                         qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                         bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                         in_features, groups, has_bias);
    }
  } else if (rows > 1 && rows <= 4 && parallel_kernel == 2) {
    hipLaunchKernelGGL((w4a4_linear_dot4_rows_kernel<4>),
                       dim3(out_features / 4, (rows + 3) / 4), dim3(128), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                       in_features, groups, has_bias);
  } else if (rows > 4 && rows <= 8 && parallel_kernel == 3) {
    hipLaunchKernelGGL((w4a4_linear_dot4_rows_kernel<8>),
                       dim3(out_features / 4, (rows + 7) / 8), dim3(256), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                       in_features, groups, has_bias);
  } else if (rows > 8 && rows <= 16 && parallel_kernel == 2) {
    hipLaunchKernelGGL((w4a4_linear_wmma_i8_kernel<16>), dim3(out_features / 16), dim3(32), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                       in_features, groups, has_bias);
  } else if (rows > 1) {
    hipLaunchKernelGGL((w4a4_linear_dot4_rows_kernel<4>),
                       dim3(out_features / 4, (rows + 3) / 4), dim3(128), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                       in_features, groups, has_bias);
  } else if (output_tile == 2 && decode_waves == 4) {
    hipLaunchKernelGGL(w4a4_linear_dot4_n2_waves4_kernel,
                       dim3(out_features / 8), dim3(128), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), out_features,
                       in_features, groups, has_bias);
  } else if (output_tile == 1) {
    hipLaunchKernelGGL((w4a4_linear_dot4_kernel<1>), dim3(out_features, rows), dim3(32), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                       in_features, groups, has_bias);
  } else if (output_tile == 2) {
    hipLaunchKernelGGL((w4a4_linear_dot4_kernel<2>), dim3(out_features / 2, rows), dim3(32), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                       in_features, groups, has_bias);
  } else if (output_tile == 4) {
    hipLaunchKernelGGL((w4a4_linear_dot4_kernel<4>), dim3(out_features / 4, rows), dim3(32), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                       in_features, groups, has_bias);
  } else if (output_tile == 8) {
    hipLaunchKernelGGL((w4a4_linear_dot4_kernel<8>), dim3(out_features / 8, rows), dim3(32), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                       in_features, groups, has_bias);
  } else if (output_tile == 16) {
    hipLaunchKernelGGL((w4a4_linear_dot4_kernel<16>), dim3(out_features / 16, rows), dim3(32), 0, 0,
                       qx.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sx.data_ptr<at::Half>()),
                       qw.data_ptr<uint8_t>(), reinterpret_cast<const half*>(sw.data_ptr<at::Half>()),
                       bias_ptr, reinterpret_cast<half*>(output.data_ptr<at::Half>()), rows, out_features,
                       in_features, groups, has_bias);
  } else {
    TORCH_CHECK(false, "output_tile must be 1, 2, 4, 8, or 16");
  }
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor embedding_lookup_w4(
    torch::Tensor token_ids,
    torch::Tensor qweight,
    torch::Tensor scales,
    int64_t group_size) {
  TORCH_CHECK(token_ids.is_cuda() && token_ids.scalar_type() == torch::kInt64, "token_ids must be HIP int64");
  TORCH_CHECK(qweight.is_cuda() && qweight.scalar_type() == torch::kUInt8 && qweight.dim() == 2,
              "qweight must be HIP uint8 [vocab, K/2]");
  check_half_cuda(scales, "embedding scales");
  TORCH_CHECK(group_size == kGroup, "only group_size=128 is supported");
  int tokens = token_ids.numel(), vocab = qweight.size(0), cols = qweight.size(1) * 2;
  int groups = cols / kGroup;
  TORCH_CHECK(scales.sizes() == torch::IntArrayRef({vocab, groups}), "invalid embedding scales");
  auto output = torch::empty({tokens, cols}, scales.options());
  hipLaunchKernelGGL(embedding_lookup_kernel, dim3(tokens, groups), dim3(kGroup), 0, 0,
                     token_ids.data_ptr<int64_t>(), qweight.data_ptr<uint8_t>(),
                     reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
                     reinterpret_cast<half*>(output.data_ptr<at::Half>()), tokens, vocab, cols, groups);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  std::vector<int64_t> shape(token_ids.sizes().begin(), token_ids.sizes().end());
  shape.push_back(cols);
  return output.view(shape);
}

std::vector<torch::Tensor> qkv_rope_hadamard(
    torch::Tensor qkv, torch::Tensor positions, double theta) {
  check_half_cuda(qkv, "qkv");
  TORCH_CHECK(qkv.dim() == 3 && qkv.size(2) == 6144, "qkv must be [batch, tokens, 6144]");
  TORCH_CHECK(positions.is_cuda() && positions.scalar_type() == torch::kInt64 && positions.is_contiguous(),
              "positions must be contiguous HIP int64");
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == qkv.size(1), "invalid positions shape");
  int batch = qkv.size(0), tokens = qkv.size(1);
  auto q = torch::empty({batch, 32, tokens, 128}, qkv.options());
  auto k = torch::empty({batch, 8, tokens, 128}, qkv.options());
  auto v = torch::empty({batch, 8, tokens, 128}, qkv.options());
  hipLaunchKernelGGL(qkv_rope_hadamard_kernel, dim3(batch * tokens * 48), dim3(128), 0, 0,
                     reinterpret_cast<const half*>(qkv.data_ptr<at::Half>()), positions.data_ptr<int64_t>(),
                     reinterpret_cast<half*>(q.data_ptr<at::Half>()),
                     reinterpret_cast<half*>(k.data_ptr<at::Half>()),
                     reinterpret_cast<half*>(v.data_ptr<at::Half>()), batch, tokens, static_cast<float>(theta));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {q, k, v};
}

torch::Tensor cross_head_hadamard(torch::Tensor attention) {
  check_half_cuda(attention, "attention");
  TORCH_CHECK(attention.dim() >= 2 && attention.size(-2) == 32 && attention.size(-1) == 128,
              "attention must end in [32, 128]");
  int vectors = attention.numel() / (32 * 128) * 128;
  auto output = torch::empty_like(attention);
  hipLaunchKernelGGL(cross_head_hadamard_kernel, dim3(vectors), dim3(32), 0, 0,
                     reinterpret_cast<const half*>(attention.data_ptr<at::Half>()),
                     reinterpret_cast<half*>(output.data_ptr<at::Half>()), vectors);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> quantize_kv_biased(
    torch::Tensor key, torch::Tensor value, double clip_ratio) {
  check_half_cuda(key, "key");
  check_half_cuda(value, "value");
  TORCH_CHECK(key.sizes() == value.sizes() && key.size(-1) == 128, "key/value shape mismatch");
  int vectors = key.numel() / 128;
  std::vector<int64_t> packed_shape(key.sizes().begin(), key.sizes().end());
  packed_shape.back() = 64;
  std::vector<int64_t> params_shape(key.sizes().begin(), key.sizes().end());
  params_shape.back() = 2;
  auto packed_key = torch::empty(packed_shape, key.options().dtype(torch::kUInt8));
  auto packed_value = torch::empty(packed_shape, key.options().dtype(torch::kUInt8));
  auto key_params = torch::empty(params_shape, key.options());
  auto value_params = torch::empty(params_shape, key.options());
  hipLaunchKernelGGL(quantize_kv_biased_kernel, dim3(vectors), dim3(128), 0, 0,
                     reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
                     reinterpret_cast<const half*>(value.data_ptr<at::Half>()),
                     packed_key.data_ptr<uint8_t>(), reinterpret_cast<half*>(key_params.data_ptr<at::Half>()),
                     packed_value.data_ptr<uint8_t>(), reinterpret_cast<half*>(value_params.data_ptr<at::Half>()),
                     vectors, static_cast<float>(clip_ratio));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {packed_key, key_params, packed_value, value_params};
}

void append_kv_biased(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cache_data,
    torch::Tensor cache_params,
    torch::Tensor page_ids,
    int64_t slot,
    int64_t page_size,
    double clip_ratio) {
  check_half_cuda(key, "key");
  check_half_cuda(value, "value");
  TORCH_CHECK(key.sizes() == value.sizes() && key.dim() == 3 && key.size(1) == 8 && key.size(2) == 128,
              "decode key/value must be [batch, 8, 128]");
  TORCH_CHECK(cache_data.is_cuda() && cache_data.scalar_type() == torch::kUInt8 && cache_data.is_contiguous(),
              "cache_data must be contiguous HIP uint8");
  check_half_cuda(cache_params, "cache_params");
  TORCH_CHECK(page_ids.is_cuda() && page_ids.scalar_type() == torch::kInt32 && page_ids.is_contiguous(),
              "page_ids must be contiguous HIP int32");
  TORCH_CHECK(page_ids.numel() == key.size(0), "page_ids must match batch");
  TORCH_CHECK(slot >= 0 && slot < page_size, "slot out of range");
  int batch = key.size(0);
  hipLaunchKernelGGL(append_kv_biased_kernel, dim3(batch * 8), dim3(128), 0, 0,
                     reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
                     reinterpret_cast<const half*>(value.data_ptr<at::Half>()), cache_data.data_ptr<uint8_t>(),
                     reinterpret_cast<half*>(cache_params.data_ptr<at::Half>()), page_ids.data_ptr<int32_t>(),
                     batch, static_cast<int>(page_size), static_cast<int>(slot), static_cast<float>(clip_ratio));
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

void append_kv_chunk_biased(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cache_data,
    torch::Tensor cache_params,
    torch::Tensor allocation_indptr,
    torch::Tensor allocation_indices,
    int64_t start_position,
    int64_t page_size,
    double clip_ratio) {
  check_half_cuda(key, "key");
  check_half_cuda(value, "value");
  TORCH_CHECK(key.sizes() == value.sizes() && key.dim() == 4 && key.size(2) == 8 && key.size(3) == 128,
              "chunk key/value must be [batch, chunk, 8, 128]");
  TORCH_CHECK(cache_data.is_cuda() && cache_data.scalar_type() == torch::kUInt8 && cache_data.is_contiguous(),
              "cache_data must be contiguous HIP uint8");
  check_half_cuda(cache_params, "cache_params");
  TORCH_CHECK(allocation_indptr.is_cuda() && allocation_indptr.scalar_type() == torch::kInt32 &&
              allocation_indptr.is_contiguous(), "allocation_indptr must be contiguous HIP int32");
  TORCH_CHECK(allocation_indices.is_cuda() && allocation_indices.scalar_type() == torch::kInt32 &&
              allocation_indices.is_contiguous(), "allocation_indices must be contiguous HIP int32");
  int batch = key.size(0), chunk = key.size(1);
  hipLaunchKernelGGL(append_kv_chunk_biased_kernel, dim3(batch * chunk * 8), dim3(128), 0, 0,
                     reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
                     reinterpret_cast<const half*>(value.data_ptr<at::Half>()), cache_data.data_ptr<uint8_t>(),
                     reinterpret_cast<half*>(cache_params.data_ptr<at::Half>()),
                     allocation_indptr.data_ptr<int32_t>(), allocation_indices.data_ptr<int32_t>(), batch, chunk,
                     static_cast<int>(page_size), static_cast<int>(start_position), static_cast<float>(clip_ratio));
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> virtual_verify_metadata(
    torch::Tensor anchor,
    int64_t batch,
    int64_t start_len,
    int64_t chunk,
    int64_t page_size,
    int64_t pages_per_batch) {
  TORCH_CHECK(anchor.is_cuda(), "anchor must be on a HIP device");
  int64_t total_entries = 0;
  for (int64_t offset = 0; offset < chunk; ++offset) {
    total_entries += (start_len + offset + 1 + page_size - 1) / page_size;
  }
  total_entries *= batch;
  auto options = anchor.options().dtype(torch::kInt32);
  auto indptr = torch::empty({batch * chunk + 1}, options);
  auto indices = torch::empty({total_entries}, options);
  auto last_page_offset = torch::empty({batch * chunk}, options);
  hipLaunchKernelGGL(virtual_verify_metadata_kernel, dim3(1), dim3(1), 0, 0,
                     indptr.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
                     last_page_offset.data_ptr<int32_t>(), static_cast<int>(batch), static_cast<int>(start_len),
                     static_cast<int>(chunk), static_cast<int>(page_size), static_cast<int>(pages_per_batch));
  C10_HIP_KERNEL_LAUNCH_CHECK();
  return {indptr, indices, last_page_offset};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_activation", &quantize_activation, "Pack symmetric group-128 activation INT4");
  m.def("rmsnorm_quantize", &rmsnorm_quantize, "Fused unweighted RMSNorm and A4 pack");
  m.def("silu_hadamard_quantize", &silu_hadamard_quantize, "Fused SiLU, H14336 and A4 pack");
  m.def("w4a4_linear", &w4a4_linear, "Direct packed W4A4 linear");
  m.def("embedding_lookup_w4", &embedding_lookup_w4, "Packed W4 embedding lookup");
  m.def("qkv_rope_hadamard", &qkv_rope_hadamard, "Fused QKV split, RoPE and H128");
  m.def("cross_head_hadamard", &cross_head_hadamard, "Fused H32 across query heads");
  m.def("quantize_kv_biased", &quantize_kv_biased, "Fused K/V biased-nibble KV4 packing");
  m.def("append_kv_biased", &append_kv_biased, "Quantize and append one token directly to KV4 pages");
  m.def("append_kv_chunk_biased", &append_kv_chunk_biased, "Quantize and append a provisional KV4 chunk");
  m.def("virtual_verify_metadata", &virtual_verify_metadata, "Build virtual verification metadata in one launch");
}
