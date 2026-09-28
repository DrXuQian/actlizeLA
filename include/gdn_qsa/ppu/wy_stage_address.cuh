#pragma once

#include "gdn_qsa/wy_contract.hpp"
#include "gdn_qsa/ppu/wy_state_address.cuh"

namespace gdn_qsa::wy {

// Preserve the original per-thread element order without signed division.
// Indices count logical matrix elements, not bytes: inverse source is FP32,
// destination is BF16, and each pointer applies its own element width.
struct PrepareElementPlan {
  static_assert(ParallelThreads == Dim);
  static constexpr unsigned InverseIterations = Chunk * Chunk / ParallelThreads;
  static constexpr unsigned ValueIterations = Chunk;
  CUTE_HOST_DEVICE static constexpr unsigned inverse(unsigned tid, unsigned iteration) {
    return tid + iteration * ParallelThreads;
  }
  CUTE_HOST_DEVICE static constexpr unsigned row(unsigned iteration) { return iteration; }
  CUTE_HOST_DEVICE static constexpr unsigned column(unsigned tid) { return tid; }
};

}  // namespace gdn_qsa::wy
