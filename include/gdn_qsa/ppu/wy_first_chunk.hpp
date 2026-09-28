#pragma once
#include <cutlass/cutlass.h>
#include "gdn_qsa/ppu/wy_full_chunk.hpp"

namespace gdn_qsa::wy::first_chunk {
// Only absence of the public state proves H0=+0. Never inspect state values.
constexpr bool eligible(int sequence, bool has_initial) {
  return !has_initial && gate_cache::full_chunks(sequence);
}
CUTLASS_HOST_DEVICE constexpr bool has_history(int chunk) { return chunk != 0; }
}  // namespace gdn_qsa::wy::first_chunk
