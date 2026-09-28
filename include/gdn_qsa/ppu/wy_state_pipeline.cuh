#pragma once

#include "gdn_qsa/ppu/wy_tiles.cuh"

namespace gdn_qsa::wy::state_pipeline {

// The pipeline reuses W only after W@H has retired. K/U remain single
// buffered: the next iteration's snapshot/W-ready barrier retires their
// previous readers before the next AIU writer. No extra storage or K split.
struct Plan {
  static constexpr int Threads = StateTile::Threads;
  static constexpr unsigned SharedBytes = sizeof(TiledStateStorage);
  CUTE_HOST_DEVICE static constexpr int next(int chunk, int chunks) {
    return chunk + 1 < chunks ? chunk + 1 : -1;
  }
  CUTE_HOST_DEVICE static constexpr int valid(int chunk, int sequence) {
    int const left = sequence - chunk * Chunk;
    return left < Chunk ? left : Chunk;
  }
};
static_assert(Plan::SharedBytes == 49408);
static_assert(Plan::Threads == 128);
static_assert(Plan::next(0, 1) == -1 && Plan::next(0, 2) == 1);
static_assert(Plan::valid(1, 65) == 1);

}  // namespace gdn_qsa::wy::state_pipeline
