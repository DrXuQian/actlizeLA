#pragma once
#include "gdn_qsa/wy_contract.hpp"

namespace gdn_qsa::wy::gate_cache {

// Host selection only. The specialized kernel never accepts a partial chunk.
constexpr bool full_chunks(int sequence) {
  return sequence > 0 && sequence % Chunk == 0;
}

}  // namespace gdn_qsa::wy::gate_cache

namespace gdn_qsa::wy {
// 固化的ZW610 shape专用选路；gate内容不参与决策，未验证shape保持回退。
enum class CoreRoute : int { Retained = 0, Solve8 = 1, Solve8Value64 = 2 };
constexpr CoreRoute measured_core_route(int batch, int sequence, int key_heads,
                                       int value_heads, bool enabled = true) {
  if (!enabled || key_heads != 16 || !gate_cache::full_chunks(sequence))
    return CoreRoute::Retained;
  if (batch == 1 && sequence == 2048 && (value_heads == 32 || value_heads == 64))
    return CoreRoute::Solve8;
  if (batch == 2 && sequence == 8192 && value_heads == 48)
    return CoreRoute::Solve8Value64;
  return CoreRoute::Retained;
}
constexpr CoreRoute default_core_route(Shape const& shape) {
#if ALA_PPU10_BEST_CORE
  return measured_core_route(shape.batch, shape.sequence, shape.q_heads, shape.value_heads);
#else
  return CoreRoute::Retained;
#endif
}
}  // namespace gdn_qsa::wy
