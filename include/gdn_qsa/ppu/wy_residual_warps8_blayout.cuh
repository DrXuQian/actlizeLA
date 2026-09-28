#pragma once

#include "gdn_qsa/ppu/wy_residual_warps8.cuh"
#include "gdn_qsa/ppu/wy_residual_blayout.cuh"

namespace gdn_qsa::wy::residual_warps8_blayout {
using residual_warps8::StateTile;
using residual_warps8::Plan;
using residual_warps8::Storage;
using residual_warps8::Key;
using residual_warps8::Inverse;
using residual_warps8::Snapshot;
using residual_warps8::Value;

// Same physical microcubes/native B reader as the four-warp experiment.
// Only the C-fragment producer changes: eight warps own one16-row value
// fragment each, not two. Derive the cube from the actual owner coordinates;
// the old (warp/2)*4 shortcut is wrong/OOB for the upper four warps.
struct BIntermediate : residual_blayout::BIntermediate {
  CUTE_HOST_DEVICE static constexpr unsigned producer_base(unsigned warp, unsigned lane) {
    unsigned const microcube = cube(StateTile::value_row(warp, 0), StateTile::column(warp));
    return microcube * 256 + (lane % 4) * 16 + lane / 4;
  }
};
static_assert(StateTile::ValueFragments == 1 && Plan::Threads == 256);
static_assert(ValueTile == 32 && sizeof(Storage) == 45568);
}  // namespace gdn_qsa::wy::residual_warps8_blayout
