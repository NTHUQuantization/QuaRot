#include <hip/hip_runtime.h>
extern "C" __global__ void rsqrt_overload_probe(const float* input, float* output) {
  float x = input[threadIdx.x];
  output[threadIdx.x * 2] = ::rsqrt(x);
  output[threadIdx.x * 2 + 1] = ::rsqrtf(x);
}
