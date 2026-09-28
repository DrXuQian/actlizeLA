#pragma once
#include "gdn_qsa/ppu/wy_residual.cuh"

namespace gdn_qsa::wy::residual_prefetch {
using residual::Key;
using residual::Inverse;
using residual::Snapshot;
using residual::Value;
using residual::Storage;
struct Plan : residual::Plan {
  CUTE_HOST_DEVICE static constexpr int next(int chunk, int chunks) {
    return chunk + 1 < chunks ? chunk + 1 : -1;
  }
};
static_assert(Plan::next(0, 1) == -1 && Plan::next(0, 2) == 1);
static_assert(Plan::Threads == 128 && sizeof(Storage) == 45568);
}  // namespace gdn_qsa::wy::residual_prefetch
