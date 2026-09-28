#pragma once
#include "gdn_qsa/wy_contract.hpp"

namespace gdn_qsa::wy::gate_cache {

// Host selection only. The specialized kernel never accepts a partial chunk.
constexpr bool full_chunks(int sequence) {
  return sequence > 0 && sequence % Chunk == 0;
}

}  // namespace gdn_qsa::wy::gate_cache
