#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace quarot_hadamard_wmma {

using f16_frag = _Float16 __attribute__((ext_vector_type(8)));
using f32_frag = float __attribute__((ext_vector_type(8)));

#if defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(HADACORE_FORCE_GFX12_WMMA)
#define QUAROT_HADACORE_GFX12_WMMA 1
#elif defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1250__))
#define QUAROT_HADACORE_GFX12_WMMA 1
#else
#define QUAROT_HADACORE_GFX12_WMMA 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(HADACORE_ENABLE_EXPERIMENTAL_WMMA) && defined(HADACORE_FORCE_GFX12_WMMA)
#define QUAROT_HADACORE_HAS_F32_F16_WMMA 1
#elif QUAROT_HADACORE_GFX12_WMMA && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
#define QUAROT_HADACORE_HAS_F32_F16_WMMA 1
#else
#define QUAROT_HADACORE_HAS_F32_F16_WMMA 0
#endif

__device__ __forceinline__ f32_frag zero_f32_frag() {
  f32_frag frag;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    frag[i] = 0.0f;
  }
  return frag;
}

__device__ __forceinline__ _Float16 f16_bits_to_f16(uint16_t x) {
  union {
    uint16_t u;
    _Float16 h;
  } v;
  v.u = x;
  return v.h;
}

__device__ __forceinline__ uint16_t f16_to_bits(_Float16 x) {
  union {
    _Float16 h;
    uint16_t u;
  } v;
  v.h = x;
  return v.u;
}

__device__ __forceinline__ uint16_t f32_to_f16_bits(float x) {
  return f16_to_bits(static_cast<_Float16>(x));
}

__device__ __forceinline__ float f16_bits_to_f32(uint16_t x) {
  return static_cast<float>(f16_bits_to_f16(x));
}

__device__ __forceinline__ f16_frag make_h16_f16_fragment(int lane, float scale) {
  f16_frag frag;
  int col = lane & 15;
  int k_base = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int k = k_base + i;
    bool positive = (__builtin_popcount(static_cast<unsigned>(k & col)) & 1) == 0;
    frag[i] = static_cast<_Float16>(positive ? scale : -scale);
  }
  return frag;
}

__device__ __forceinline__ f32_frag wmma_h16_f16_f32(f16_frag a, f16_frag h) {
  f32_frag c = zero_f32_frag();
#if QUAROT_HADACORE_HAS_F32_F16_WMMA
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, h, c);
#else
  return c;
#endif
}

__device__ __forceinline__ f16_frag load_f16_rowmajor_16x16(
    const uint16_t* base,
    int stride,
    int lane) {
  f16_frag frag;
  int row = lane & 15;
  int k_base = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    frag[i] = f16_bits_to_f16(base[row * stride + k_base + i]);
  }
  return frag;
}

__device__ __forceinline__ f16_frag load_f16_transposed_16x16(const uint16_t* base, int lane) {
  f16_frag frag;
  int row = lane & 15;
  int k_base = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    frag[i] = f16_bits_to_f16(base[(k_base + i) * 16 + row]);
  }
  return frag;
}

__device__ __forceinline__ void store_f16_rowmajor_16x16(
    uint16_t* base,
    int stride,
    int lane,
    f32_frag frag) {
  int col = lane & 15;
  int row_base = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    base[(row_base + i) * stride + col] = f32_to_f16_bits(frag[i]);
  }
}

__device__ __forceinline__ void store_f16_transposed_16x16(
    uint16_t* base,
    int lane,
    f32_frag frag) {
  int col = lane & 15;
  int row_base = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    base[col * 16 + row_base + i] = f32_to_f16_bits(frag[i]);
  }
}

__device__ __forceinline__ void h256_f16(
    const uint16_t* in,
    uint16_t* temp,
    uint16_t* out,
    int lane) {
  f16_frag h = make_h16_f16_fragment(lane, 0.25f);
  f32_frag d0 = wmma_h16_f16_f32(load_f16_rowmajor_16x16(in, 16, lane), h);
  store_f16_rowmajor_16x16(temp, 16, lane, d0);
  __syncthreads();
  f32_frag d1 = wmma_h16_f16_f32(load_f16_transposed_16x16(temp, lane), h);
  store_f16_transposed_16x16(out, lane, d1);
  __syncthreads();
}

__device__ __forceinline__ f16_frag load_outer_f16(const uint16_t* h256, int low_base, int lane) {
  f16_frag frag;
  int low_off = lane & 15;
  int k_base = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int chunk = k_base + i;
    frag[i] = f16_bits_to_f16(h256[chunk * 256 + low_base + low_off]);
  }
  return frag;
}

__device__ __forceinline__ void store_outer_f16(
    uint16_t* out,
    int low_base,
    int lane,
    f32_frag frag) {
  int chunk_out = lane & 15;
  int low_row = (lane >> 4) * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[chunk_out * 256 + low_base + low_row + i] = f32_to_f16_bits(frag[i]);
  }
}

}  // namespace quarot_hadamard_wmma
