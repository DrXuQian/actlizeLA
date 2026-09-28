#pragma once

#include "gdn_qsa/wy_contract.hpp"
#include "gdn_qsa/ppu/wy_mma.cuh"

namespace gdn_qsa::wy {

// Used by real device producers/consumers AND the exhaustive native-traits
// ownership gate. Four warps split independent output rows, never reduction K.
struct StateTile {
  static constexpr int Threads = 128;
  static constexpr int KFragments = 4;
  static constexpr int ValueFragments = 2;
  CUTE_HOST_DEVICE static constexpr int k_row(int warp, int fragment) {
    return (warp / 2) * 64 + fragment * 16;
  }
  CUTE_HOST_DEVICE static constexpr int value_row(int warp, int fragment) {
    return (warp / 2) * 32 + fragment * 16;
  }
  CUTE_HOST_DEVICE static constexpr int column(int warp) { return (warp % 2) * 16; }
};

struct OutputTile {
  static constexpr int Threads = 256;
  static constexpr int Panel = 64;
  static constexpr int Fragments = 2;
  CUTE_HOST_DEVICE static constexpr int row(int warp) { return (warp / 2) * 16; }
  CUTE_HOST_DEVICE static constexpr int column(int warp, int fragment) {
    return (warp % 2) * 32 + fragment * 16;
  }
};

struct PrepareTile {
  static constexpr int Threads = 128;
  static constexpr int Panel = 64;
  static constexpr int Fragments = 4;
  CUTE_HOST_DEVICE static constexpr int row(int warp) { return warp * 16; }
  CUTE_HOST_DEVICE static constexpr int column(int fragment) { return fragment * 16; }
};

struct TiledStateStorage {
  union {
    alignas(128) BF16 w[Chunk * Dim];
    alignas(128) float final_h[Dim * ValueTile];
  };
  alignas(128) BF16 k[Chunk * Dim];
  // H must remain live while U is prefetched and consumed.
  alignas(128) BF16 snapshot[Dim * ValueTile];
  alignas(128) BF16 u[Chunk * ValueTile], scaled_v[Chunk * ValueTile];
  float g[Chunk];
};
static_assert(sizeof(TiledStateStorage) == 49408);

struct TiledOutputStorage {
  alignas(128) BF16 q[Chunk * Dim], attn[Chunk * Chunk];
  union {
    alignas(128) BF16 k[Chunk * Dim];
    alignas(128) BF16 h[Dim * OutputTile::Panel];
    alignas(128) BF16 output[Chunk * OutputTile::Panel];
  } stage;
  alignas(128) BF16 v[Chunk * OutputTile::Panel];
  float g[Chunk];
};
static_assert(sizeof(TiledOutputStorage) == 49408);

static_assert(StateTile::Threads / 32 * StateTile::KFragments * 16 * 16 == Dim * ValueTile);
static_assert(StateTile::Threads / 32 * StateTile::ValueFragments * 16 * 16 == Chunk * ValueTile);
static_assert(OutputTile::Threads / 32 * OutputTile::Fragments * 16 * 16 == Chunk * OutputTile::Panel);
static_assert(PrepareTile::Threads / 32 * PrepareTile::Fragments * 16 * 16 == Chunk * PrepareTile::Panel);
}  // namespace gdn_qsa::wy
