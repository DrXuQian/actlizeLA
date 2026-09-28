// Compile-only attribution, not a numerical test or production entrypoint.
// Within each load family, change ONLY whether the cube base varies by warp.
// The two families have different shared-layout contracts: this input does
// NOT establish cross-family numerical equivalence or an AIU delivery test.
#include <hggc_runtime.h>
#include <hggc_mma.h>
#include <cute/arch/copy_ppu0010_aiu.hpp>
#include <cutlass/bfloat16.h>

template <bool Ncom, bool WarpBase>
__global__ void address_class_probe(unsigned const* input, unsigned* output) {
  __shared__ __align__(128) cutlass::bfloat16_t tiles[1024];
  unsigned const tid = unsigned(threadIdx.x);
  unsigned const lane = tid & 31u, warp = tid >> 5;
  for (unsigned i = tid; i < 512; i += 128)
    reinterpret_cast<unsigned*>(tiles)[i] = input[i];
  __syncthreads();

  unsigned const cube = WarpBase ? warp : 0;
  auto* base = tiles + cube * 256;
  unsigned reg[4];
  if constexpr (Ncom) {
    unsigned const row = (lane & 7u) + (lane >> 4) * 8u;
    unsigned const col = ((lane >> 3) & 1u) * 8u;
    awmma::ldmatrix<awmma::no_trans, 4>(reg, base + row * 16 + col);
  } else {
    cute::PPU0010_TSM_LD_SWZL<cutlass::bfloat16_t, 16, 16, true, false>::copy(
        reg, base, 0, 0);
  }
  for (unsigned i = 0; i < 4; ++i) output[tid * 4 + i] = reg[i];
}

extern "C" void instantiate_address_class(unsigned const* input, unsigned* output) {
  address_class_probe<false, false><<<1, 128>>>(input, output);
  address_class_probe<false, true><<<1, 128>>>(input, output);
  address_class_probe<true, false><<<1, 128>>>(input, output);
  address_class_probe<true, true><<<1, 128>>>(input, output);
}
