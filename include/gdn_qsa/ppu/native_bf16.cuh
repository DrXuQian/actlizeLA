#pragma once
#include <cutlass/bfloat16.h>

namespace gdn_qsa::wy::native_bf16 {
// Same scalar rounding boundary and storage type as BF16(float). Do not
// redefine HGCC version macros: their major is not a CUDA major version.
// Special-value/subnormal semantics are admitted by the separate raw-bit
// device probe before any performance capture, not inferred from compilation.
CUTLASS_DEVICE cutlass::bfloat16_t convert(float value) {
  uint16_t bits;
  asm("ppu.cvt.rtte.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
  return cutlass::bfloat16_t::bitcast(bits);
}
}  // namespace gdn_qsa::wy::native_bf16
