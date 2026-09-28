#pragma once
#include "gdn_qsa/ppu/wy_residual.cuh"

namespace gdn_qsa::wy::residual_warps8 {
// Preserve the entire V32 CTA tile and every shared layout. Only independent
// output-row ownership changes; no warp splits a reduction dimension.
using residual::Key;
using residual::Inverse;
using residual::Snapshot;
using residual::Value;
using residual::Storage;

struct StateTile {
  static constexpr int Threads = 256;
  static constexpr int KFragments = 2;
  static constexpr int ValueFragments = 1;
  CUTE_HOST_DEVICE static constexpr int k_row(int warp, int fragment) {
    return (warp / 2) * 32 + fragment * 16;
  }
  CUTE_HOST_DEVICE static constexpr int value_row(int warp, int) {
    return (warp / 2) * 16;
  }
  CUTE_HOST_DEVICE static constexpr int column(int warp) {
    return (warp % 2) * 16;
  }
};
struct Plan : residual::Plan {
  static constexpr int Threads = StateTile::Threads;
};
static_assert(ValueTile == 32 && sizeof(Storage) == 45568 && Plan::Threads == 256);
static_assert(StateTile::Threads / 32 * StateTile::KFragments * 256 == Dim * ValueTile);
static_assert(StateTile::Threads / 32 * StateTile::ValueFragments * 256 == Chunk * ValueTile);
}  // namespace gdn_qsa::wy::residual_warps8
