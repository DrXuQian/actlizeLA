#pragma once
#include <cmath>
#include <cstdint>
#if defined(ALA_PPU_TARGET) && defined(__HGGC_ARCH__) &&                       \
    __HGGC_ARCH__ != ALA_PPU_TARGET
#error "serving object compiled for a different PPU generation"
#endif
#if !defined(ALA_PPU_TARGET) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1200
#error "serving NVIDIA validation binary must target SM120, not SM90a"
#endif
#if defined(ALA_PPU_TARGET)
#include <hggc_runtime.h>
namespace actlize_la::serving {
using Stream = hggcStream_t;
inline int last_error() { return int(hggcGetLastError()); }
} // namespace actlize_la::serving
#else
#include <cuda_runtime.h>
namespace actlize_la::serving {
using Stream = cudaStream_t;
inline int last_error() { return int(cudaGetLastError()); }
} // namespace actlize_la::serving
#endif

namespace actlize_la::serving {
// BF16 storage without a CUTLASS target dependency. RNE, including NaN
// preservation; this is not a half conversion or an SDK header substitute.
using BFloat = uint16_t;
__device__ __forceinline__ float bf(BFloat x) {
  return __uint_as_float(uint32_t(x) << 16);
}
__device__ __forceinline__ BFloat bf(float x) {
  uint32_t b = __float_as_uint(x);
  if ((b & 0x7fffffffU) > 0x7f800000U)
    return BFloat((b >> 16) | 0x40);
  return BFloat((b + 0x7fffU + ((b >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 16; d; d /= 2)
    x += __shfl_xor_sync(0xffffffffU, x, d);
  return x;
}
__device__ __forceinline__ float read_bf(void const *p, int64_t i) {
  return bf(static_cast<BFloat const *>(p)[i]);
}
__device__ __forceinline__ float round_bf(float x) { return bf(bf(x)); }
__device__ __forceinline__ float log_decay(float a, float a_log, float bias) {
  float const z = a + bias;
  // Avoid an overflowing exp on the branch where softplus is exactly z.
  return -expf(a_log) * (z > 20.0f ? z : log1pf(expf(z)));
}
} // namespace actlize_la::serving
