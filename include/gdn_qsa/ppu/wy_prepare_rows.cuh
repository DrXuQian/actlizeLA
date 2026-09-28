#pragma once

#include "gdn_qsa/wy_contract.hpp"
#include <cute/config.hpp>

namespace gdn_qsa::wy {

struct PrepareRowsPlan {
  static constexpr unsigned Warp = 32;
  static constexpr unsigned Slots = 2;
  static constexpr unsigned FullMask = 0xffffffffu;
  static_assert(Chunk == Warp * Slots && ParallelThreads == 4 * Warp);
  CUTE_HOST_DEVICE static constexpr bool shared_writer(unsigned tid) { return tid < Chunk; }
  CUTE_HOST_DEVICE static constexpr unsigned producer_row(unsigned lane, unsigned slot) {
    return lane + slot * Warp;
  }
  CUTE_HOST_DEVICE static constexpr unsigned slot(unsigned row) { return row / Warp; }
  CUTE_HOST_DEVICE static constexpr unsigned lane(unsigned row) { return row % Warp; }
};

}  // namespace gdn_qsa::wy
