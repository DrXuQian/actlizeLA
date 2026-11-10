#pragma once
#include <cutlass/bfloat16.h>

namespace gdn_qsa::wy {

// Keep each original scalar rounding boundary. HGCC's major version is not
// a CUDA major; do not change global version macros to enable one conversion.
template <bool Native>
CUTLASS_DEVICE cutlass::bfloat16_t convert_bf16(float value) {
  if constexpr (Native) {
    uint16_t bits;
    asm("ppu.cvt.rtte.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
    return cutlass::bfloat16_t::bitcast(bits);
  } else {
    return cutlass::bfloat16_t(value);
  }
}

}  // namespace gdn_qsa::wy
