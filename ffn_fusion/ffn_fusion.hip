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

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__device__ __forceinline__ uint8_t pack_s4_pair(int lo, int hi) {
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
  extern __shared__ float smem[];
  float* values = smem;
  float* reduce = smem + GroupSize;

  int tid = threadIdx.x;
  int row = blockIdx.x;
  int group = blockIdx.y;
  int base = row * cols + group * GroupSize;

  float g = __half2float(gate[base + tid]);
  float u = __half2float(up[base + tid]);
  values[tid] = silu(g) * u;
  __syncthreads();

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

  float scale = fmaxf(reduce[0] / 7.0f, 1.0e-8f);
  if (tid == 0) {
    scales[row * gridDim.y + group] = __float2half(scale);
  }

  if ((tid & 1) == 0) {
    int q0 = static_cast<int>(nearbyintf(values[tid] / scale));
    int q1 = static_cast<int>(nearbyintf(values[tid + 1] / scale));
    q0 = max(-8, min(7, q0));
    q1 = max(-8, min(7, q1));
    packed[(row * cols + group * GroupSize + tid) >> 1] = pack_s4_pair(q0, q1);
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

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "fused_ffn_silu_hadamard_quant",
      &fused_ffn_silu_hadamard_quant,
      pybind11::arg("gate"),
      pybind11::arg("up"),
      pybind11::arg("group_size") = 256);
}
