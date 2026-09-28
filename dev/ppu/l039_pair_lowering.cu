// Compile-only probe. A source bf16x2 mnemonic is not a native speed claim.
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_conversion.h>
#include <hggc_bf16.h>

using BF16 = cutlass::bfloat16_t;
using Pair = cutlass::Array<BF16,2>;

template<int Mode>
__device__ __forceinline__ Pair convert(float a, float b) {
  if constexpr (Mode == 0) {
    Pair out; out[0] = BF16(a); out[1] = BF16(b); return out;
  } else if constexpr (Mode == 1) {
    cutlass::Array<float,2> in; in[0] = a; in[1] = b;
    return cutlass::NumericArrayConverter<BF16,float,2>{}(in);
  } else {
    auto packed = __floats2bfloat162_rn(a,b);
    return *reinterpret_cast<Pair const*>(&packed);
  }
}

// Same inputs and logical BF16 values, two destination contracts. Scattered
// models two NewV C slots; packed is a capability control, NOT its layout.
template<int Mode, bool Packed>
__device__ __forceinline__ void publish(float const* src, BF16* dst) {
  unsigned const tid = threadIdx.x;
  Pair const converted = convert<Mode>(src[2*tid], src[2*tid+1]);
  if constexpr (Packed) {
    *reinterpret_cast<unsigned*>(dst + 2*tid) =
        *reinterpret_cast<unsigned const*>(&converted);
  } else {
    dst[tid] = converted[0];
    dst[tid+256] = converted[1];
  }
}

extern "C" __global__ void scalar_scattered(float const* in, BF16* out) { publish<0,false>(in,out); }
extern "C" __global__ void actlize_scattered(float const* in, BF16* out) { publish<1,false>(in,out); }
extern "C" __global__ void sdk_scattered(float const* in, BF16* out) { publish<2,false>(in,out); }
extern "C" __global__ void scalar_packed(float const* in, BF16* out) { publish<0,true>(in,out); }
extern "C" __global__ void actlize_packed(float const* in, BF16* out) { publish<1,true>(in,out); }
extern "C" __global__ void sdk_packed(float const* in, BF16* out) { publish<2,true>(in,out); }
