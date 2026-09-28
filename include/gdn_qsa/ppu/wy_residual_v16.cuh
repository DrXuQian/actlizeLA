#pragma once
#include "gdn_qsa/ppu/wy_residual.cuh"

namespace gdn_qsa::wy::residual_v16 {
constexpr int ValueTile = 16;
using Key = residual::Key;
using Inverse = residual::Inverse;
using Snapshot = aiu::Tile<Dim, ValueTile>;
using Value = aiu::Tile<Chunk, ValueTile>;
struct StateTile {
  static constexpr int Threads = 128, KFragments = 2, ValueFragments = 1;
  CUTE_HOST_DEVICE static constexpr int k_row(int warp, int fragment) { return warp * 32 + fragment * 16; }
  CUTE_HOST_DEVICE static constexpr int value_row(int warp, int) { return warp * 16; }
  CUTE_HOST_DEVICE static constexpr int column(int) { return 0; }
};
struct Plan : residual::Plan {};
struct Storage {
  union {
    alignas(128) BF16 k[Chunk * Dim];
    alignas(128) float final_h[Dim * ValueTile];
  };
  alignas(128) BF16 inverse[Chunk * Chunk];
  alignas(128) BF16 snapshot[Dim * ValueTile];
  alignas(128) BF16 value[Chunk * ValueTile], residual[Chunk * ValueTile], scaled[Chunk * ValueTile];
  float gates[Chunk], beta[Chunk];
};
static_assert(sizeof(Storage) == 35328 && Plan::Threads == StateTile::Threads);
static_assert(4 * StateTile::KFragments * 256 == Dim * ValueTile);
static_assert(4 * StateTile::ValueFragments * 256 == Chunk * ValueTile);
}  // namespace gdn_qsa::wy::residual_v16
