// Compile-only SDK reachability probe; NOT a numerical/performance test.
// No SDK implementation is vendored: use its documented public header/API.
#include <hggc_runtime.h>
#include <hggc_mma.h>

template <bool Transpose>
__global__ void plain_load_probe(unsigned const* input, unsigned* output) {
  __shared__ __align__(16) unsigned tile[128];
  unsigned const lane = unsigned(threadIdx.x) & 31u;
  for (unsigned i = lane; i < 128; i += 32) tile[i] = input[i];
  __syncthreads();
  unsigned reg[4];
  auto const* address = tile + (lane & 7u) * 8u + (lane >> 4) * 64u + ((lane >> 3) & 1u) * 4u;
  if constexpr (Transpose)
    awmma::ldmatrix<awmma::trans_16x16b16, 4>(reg, address);
  else
    awmma::ldmatrix<awmma::no_trans, 4>(reg, address);
  for (unsigned i = 0; i < 4; ++i) output[lane * 4 + i] = reg[i];
}

extern "C" void instantiate_plain_load_probe(unsigned const* input, unsigned* output) {
  plain_load_probe<false><<<1, 32>>>(input, output);
  plain_load_probe<true><<<1, 32>>>(input, output);
}
