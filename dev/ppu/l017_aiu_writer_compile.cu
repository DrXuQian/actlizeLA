// Compile-only attribution of the native AIU SWZL/linear writer bit.
// Scalar reads keep the transfer live; no numerical/MMA or timing claim.
#include <hggc_runtime.h>
#include <cute/arch/copy_ppu0010_aiu.hpp>
#include <cutlass/bfloat16.h>

template <bool Swzl>
__global__ void aiu_writer_probe(cutlass::bfloat16_t const* input, unsigned* output) {
  __shared__ __align__(128) cutlass::bfloat16_t sm[4096];
  if (threadIdx.x == 0) {
    cute::AiuDesc desc{};
    desc.dim_h = 64; desc.dim_w = 64;
    desc.cube_h = 64; desc.cube_w = 64;
    cute::PPU0010_AIU_LOAD<cute::Int<65536>, cutlass::bfloat16_t, false, Swzl>::copy(
        sm, input, desc, 0, 0);
  }
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();
  output[threadIdx.x] = sm[threadIdx.x].raw();
}

extern "C" void instantiate_aiu_writer(cutlass::bfloat16_t const* input, unsigned* output) {
  aiu_writer_probe<true><<<1, 128>>>(input, output);
  aiu_writer_probe<false><<<1, 128>>>(input, output);
}
