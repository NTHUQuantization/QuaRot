#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "../hadamard_wmma.cuh"

#define HIP_ASSERT(x) TORCH_CHECK((x) == hipSuccess, "HIP error: ", hipGetErrorString(x))

namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ uint8_t pack_s4_pair(int lo, int hi) {
  // INT4 is stored as unsigned nibbles with an offset of +8:
  // logical range [-8, 7] becomes storage range [0, 15].
  uint8_t ulo = static_cast<uint8_t>(lo + 8) & 0x0f;
  uint8_t uhi = static_cast<uint8_t>(hi + 8) & 0x0f;
  return static_cast<uint8_t>(ulo | (uhi << 4));
}

template <int GroupSize>
__global__ void fused_ffn_silu_hadamard_quant_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    uint8_t* __restrict__ packed,
    half* __restrict__ scales,
    int rows,
    int cols) {
  // One block owns one contiguous FFN quantization group:
  //   blockIdx.x = flattened row, blockIdx.y = group along the hidden dim.
  // Each thread owns one element inside the group, so the whole fused path
  // stays in shared memory until the final INT4 write.
  extern __shared__ float smem[];
  float* values = smem;
  float* reduce = smem + GroupSize;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int group = blockIdx.y;
  int base = row * cols + group * GroupSize;

  float g = __half2float(gate[base + tid]);
  float u = __half2float(up[base + tid]);
  // FFN activation for SwiGLU-style MLP: silu(gate) * up.
  values[tid] = silu(g) * u;
  __syncthreads();

  // In-place fast Walsh-Hadamard transform. The butterfly index maps each
  // active thread to the lower element of a pair separated by "stride".
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

  // Orthonormal Hadamard: divide by sqrt(GroupSize). QuaRot uses this rotation
  // before quantization to spread outliers across the group.
  const float norm = rsqrtf(static_cast<float>(GroupSize));
  float rotated = values[tid] * norm;
  values[tid] = rotated;
  reduce[tid] = fabsf(rotated);
  __syncthreads();

  for (int stride = GroupSize >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
    }
    __syncthreads();
  }

  // Symmetric dynamic INT4 scale for the whole group. Dividing by 7 preserves
  // the positive endpoint; values are later clamped into [-8, 7].
  float scale = fmaxf(reduce[0] / 7.0f, 1.0e-8f);
  if (tid == 0) {
    scales[row * gridDim.y + group] = __float2half(scale);
  }

  // Two signed INT4 values are packed into one byte: even thread writes the
  // pair (tid, tid + 1), low nibble first.
  if ((tid & 1) == 0) {
    int q0 = static_cast<int>(nearbyintf(values[tid] / scale));
    int q1 = static_cast<int>(nearbyintf(values[tid + 1] / scale));
    q0 = max(-8, min(7, q0));
    q1 = max(-8, min(7, q1));
    packed[(row * cols + group * GroupSize + tid) >> 1] = pack_s4_pair(q0, q1);
  }
}

__global__ void fused_ffn_silu_hadamard_quant_hadacore256_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    uint8_t* __restrict__ packed,
    half* __restrict__ scales,
    int rows,
    int cols) {
  // Hadacore path fixes the group size at 256 and maps the group to one wave
  // of 32 threads. Each lane processes 8 elements before/after the WMMA helper.
  __shared__ uint16_t in_bits[256];
  __shared__ uint16_t temp_bits[256];
  __shared__ uint16_t out_bits[256];
  __shared__ float reduce[32];

  int lane = threadIdx.x & 31;
  int row = blockIdx.x;
  int group = blockIdx.y;
  int base = row * cols + group * 256;

  for (int i = lane; i < 256; i += 32) {
    float g = __half2float(gate[base + i]);
    float u = __half2float(up[base + i]);
    in_bits[i] = quarot_hadamard_wmma::f32_to_f16_bits(silu(g) * u);
  }
  __syncthreads();

#if QUAROT_HADACORE_HAS_F32_F16_WMMA
  // Hardware-accelerated 256-point Hadamard. The helper also applies the
  // 1/sqrt(256) normalization internally for the f16 path.
  quarot_hadamard_wmma::h256_f16(in_bits, temp_bits, out_bits, lane);
#else
  // Portable fallback used when the WMMA implementation is unavailable.
  if (lane == 0) {
    for (int stride = 1; stride < 256; stride <<= 1) {
      for (int p = 0; p < 128; ++p) {
        int q = p & (stride - 1);
        int idx = ((p - q) << 1) + q;
        float a = quarot_hadamard_wmma::f16_bits_to_f32(in_bits[idx]);
        float b = quarot_hadamard_wmma::f16_bits_to_f32(in_bits[idx + stride]);
        temp_bits[idx] = quarot_hadamard_wmma::f32_to_f16_bits(a + b);
        temp_bits[idx + stride] = quarot_hadamard_wmma::f32_to_f16_bits(a - b);
      }
      for (int i = 0; i < 256; ++i) {
        in_bits[i] = temp_bits[i];
      }
    }
    for (int i = 0; i < 256; ++i) {
      out_bits[i] = quarot_hadamard_wmma::f32_to_f16_bits(
          quarot_hadamard_wmma::f16_bits_to_f32(in_bits[i]) * 0.0625f);
    }
  }
  __syncthreads();
#endif

  float local_max = 0.0f;
  for (int i = lane; i < 256; i += 32) {
    local_max = fmaxf(local_max, fabsf(quarot_hadamard_wmma::f16_bits_to_f32(out_bits[i])));
  }
  reduce[lane] = local_max;
  __syncthreads();

  for (int stride = 16; stride > 0; stride >>= 1) {
    if (lane < stride) {
      reduce[lane] = fmaxf(reduce[lane], reduce[lane + stride]);
    }
    __syncthreads();
  }

  float scale = fmaxf(reduce[0] / 7.0f, 1.0e-8f);
  if (lane == 0) {
    scales[row * gridDim.y + group] = __float2half(scale);
  }

  for (int pair = lane; pair < 128; pair += 32) {
    float x0 = quarot_hadamard_wmma::f16_bits_to_f32(out_bits[pair << 1]);
    float x1 = quarot_hadamard_wmma::f16_bits_to_f32(out_bits[(pair << 1) + 1]);
    int q0 = max(-8, min(7, static_cast<int>(nearbyintf(x0 / scale))));
    int q1 = max(-8, min(7, static_cast<int>(nearbyintf(x1 / scale))));
    packed[(row * cols + group * 256 + (pair << 1)) >> 1] = pack_s4_pair(q0, q1);
  }
}

template <int GroupSize>
void launch_fused(torch::Tensor gate, torch::Tensor up, torch::Tensor packed, torch::Tensor scales) {
  int rows = static_cast<int>(gate.numel() / gate.size(-1));
  int cols = static_cast<int>(gate.size(-1));
  dim3 grid(rows, cols / GroupSize);
  dim3 block(GroupSize);
  size_t smem = static_cast<size_t>(GroupSize) * 2 * sizeof(float);
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();

  hipLaunchKernelGGL(
      (fused_ffn_silu_hadamard_quant_kernel<GroupSize>),
      grid,
      block,
      smem,
      stream,
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
      rows,
      cols);
  HIP_ASSERT(hipGetLastError());
}

void launch_hadacore256(torch::Tensor gate, torch::Tensor up, torch::Tensor packed, torch::Tensor scales) {
  int rows = static_cast<int>(gate.numel() / gate.size(-1));
  int cols = static_cast<int>(gate.size(-1));
  dim3 grid(rows, cols / 256);
  dim3 block(32);
  hipStream_t stream = at::hip::getCurrentHIPStream().stream();

  hipLaunchKernelGGL(
      fused_ffn_silu_hadamard_quant_hadacore256_kernel,
      grid,
      block,
      0,
      stream,
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
      rows,
      cols);
  HIP_ASSERT(hipGetLastError());
}

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant(
    torch::Tensor gate,
    torch::Tensor up,
    int64_t group_size) {
  TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA/HIP tensor");
  TORCH_CHECK(up.is_cuda(), "up must be a CUDA/HIP tensor");
  TORCH_CHECK(gate.scalar_type() == torch::kFloat16, "gate must be float16");
  TORCH_CHECK(up.scalar_type() == torch::kFloat16, "up must be float16");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
  TORCH_CHECK(gate.dim() >= 1, "gate must have at least one dimension");
  TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
  TORCH_CHECK(up.is_contiguous(), "up must be contiguous");

  int64_t cols = gate.size(-1);
  TORCH_CHECK(group_size == 128 || group_size == 256 || group_size == 512 || group_size == 1024,
              "group_size must be one of 128, 256, 512, 1024");
  TORCH_CHECK(cols % group_size == 0, "last dimension must be divisible by group_size");
  TORCH_CHECK((cols % 2) == 0, "last dimension must be even for int4 packing");

  auto packed_shape = gate.sizes().vec();
  packed_shape.back() = cols / 2;
  auto scale_shape = gate.sizes().vec();
  scale_shape.back() = cols / group_size;

  auto packed = torch::empty(packed_shape, gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, gate.options());

  at::hip::HIPGuard device_guard{static_cast<char>(gate.get_device())};
  switch (group_size) {
    case 128:
      launch_fused<128>(gate, up, packed, scales);
      break;
    case 256:
      launch_fused<256>(gate, up, packed, scales);
      break;
    case 512:
      launch_fused<512>(gate, up, packed, scales);
      break;
    case 1024:
      launch_fused<1024>(gate, up, packed, scales);
      break;
  }

  return {packed, scales};
}

std::vector<torch::Tensor> fused_ffn_silu_hadamard_quant_hadacore256(
    torch::Tensor gate,
    torch::Tensor up) {
  TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA/HIP tensor");
  TORCH_CHECK(up.is_cuda(), "up must be a CUDA/HIP tensor");
  TORCH_CHECK(gate.scalar_type() == torch::kFloat16, "gate must be float16");
  TORCH_CHECK(up.scalar_type() == torch::kFloat16, "up must be float16");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
  TORCH_CHECK(gate.dim() >= 1, "gate must have at least one dimension");
  TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
  TORCH_CHECK(up.is_contiguous(), "up must be contiguous");

  int64_t cols = gate.size(-1);
  TORCH_CHECK(cols % 256 == 0, "last dimension must be divisible by 256");
  TORCH_CHECK((cols % 2) == 0, "last dimension must be even for int4 packing");

  auto packed_shape = gate.sizes().vec();
  packed_shape.back() = cols / 2;
  auto scale_shape = gate.sizes().vec();
  scale_shape.back() = cols / 256;

  auto packed = torch::empty(packed_shape, gate.options().dtype(torch::kUInt8));
  auto scales = torch::empty(scale_shape, gate.options());

  at::hip::HIPGuard device_guard{static_cast<char>(gate.get_device())};
  launch_hadacore256(gate, up, packed, scales);
  return {packed, scales};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fused_ffn_silu_hadamard_quant",
      &fused_ffn_silu_hadamard_quant,
      pybind11::arg("gate"),
      pybind11::arg("up"),
      pybind11::arg("group_size") = 256);
  m.def(
      "fused_ffn_silu_hadamard_quant_hadacore256",
      &fused_ffn_silu_hadamard_quant_hadacore256,
      pybind11::arg("gate"),
      pybind11::arg("up"));
}
