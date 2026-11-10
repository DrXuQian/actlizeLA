// Compile-only attribution probe; not part of the product build or a benchmark.
#include <cutlass/bfloat16.h>
#define ALA_STR_INNER(x) #x
#define ALA_STR(x) ALA_STR_INNER(x)
#if defined(__HGGC_ARCH__)
#pragma message("GDN_BF16_PROBE __HGGC_ARCH__=" ALA_STR(__HGGC_ARCH__))
#pragma message("GDN_BF16_PROBE __HGGCCC_VER_MAJOR__=" ALA_STR(__HGGCCC_VER_MAJOR__))
#if (__HGGC_ARCH__ >= 100) && (__HGGCCC_VER_MAJOR__ >= 11)
#pragma message("GDN_BF16_PROBE cutlass_constructor=NATIVE")
#else
#pragma message("GDN_BF16_PROBE cutlass_constructor=SOFTWARE")
#endif
#endif

extern "C" __global__ void bf16_ctor(float const* input, unsigned short* out) {
  int i = int(threadIdx.x) + int(blockIdx.x) * int(blockDim.x);
  out[i] = cutlass::bfloat16_t(input[i]).raw();
}

extern "C" __global__ void bf16_native(float const* input, unsigned short* out) {
  int i = int(threadIdx.x) + int(blockIdx.x) * int(blockDim.x);
  unsigned short bits;
  asm("ppu.cvt.rtte.bf16.f32 %0, %1;" : "=h"(bits) : "f"(input[i]));
  out[i] = bits;
}
