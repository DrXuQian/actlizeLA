#pragma once

#include "gdn_qsa/ppu/wy_aiu.cuh"
#include "gdn_qsa/ppu/wy_tiles.cuh"

namespace gdn_qsa::wy::residual {
// This is a new arithmetic contract, NOT a raw-bit delivery variant.
// P[BV - BD(KH)] replaces PBV - (PBDK)H. P and H are BF16;
// residual, Vnew and scaledV have explicit BF16 boundaries; H stays FP32.
using Key = aiu::Tile<Chunk, Dim>;
using Inverse = aiu::Tile<Chunk, Chunk>;
using Snapshot = aiu::Tile<Dim, ValueTile>;
using Value = aiu::Tile<Chunk, ValueTile>;

struct Plan {
  static constexpr int Threads = StateTile::Threads;
  // Match the UNCHANGED split-solve publisher. This is a padded BF16-element
  // pitch, not the compact C*C extent. Do not alias it with H snapshots:
  // independent V-slice CTAs can overwrite H before another CTA reads P.
  CUTE_HOST_DEVICE static constexpr int64_t inverse_base(int64_t group) {
    return state_offset(group);
  }
  CUTE_HOST_DEVICE static constexpr int valid(int ct, int sequence) {
    int const remaining = sequence - ct * Chunk;
    return remaining < Chunk ? remaining : Chunk;
  }
};

struct Storage {
  union {
    alignas(128) BF16 k[Chunk * Dim];
    alignas(128) float final_h[Dim * ValueTile];
  };
  alignas(128) BF16 inverse[Chunk * Chunk];
  alignas(128) BF16 snapshot[Dim * ValueTile];
  alignas(128) BF16 value[Chunk * ValueTile];
  alignas(128) BF16 residual[Chunk * ValueTile];
  alignas(128) BF16 scaled[Chunk * ValueTile];
  float gates[Chunk], beta[Chunk];
};
static_assert(sizeof(Storage) == 45568);
static_assert(Plan::Threads == 128 && Plan::inverse_base(1) == 16384);
}  // namespace gdn_qsa::wy::residual
