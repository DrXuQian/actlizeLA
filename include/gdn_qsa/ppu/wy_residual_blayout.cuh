#pragma once

#include "gdn_qsa/ppu/wy_residual.cuh"

namespace gdn_qsa::wy::residual_blayout {
using residual::Plan;
using residual::Storage;
using residual::Key;
using residual::Inverse;
using residual::Snapshot;
using residual::Value;

// Only register-produced, non-published B operands use this orientation.
// Logical (time,value) is stored as physical (value,time), directly at the
// existing BF16 conversion/store. No copy/transpose/shuffle is inserted.
// AIU inputs and the globally published H/Vnew retain their original layout.
struct BIntermediate {
  // Microcube swizzle can be folded into a lane base + immediate per C slot.
  // A wider AIU cube has cross-slice rotations: the first implementation
  // compiled to extra per-slot address arithmetic and was rejected locally.
  using Physical = aiu::Tile<16, 16>;
  using Layout = decltype(cute::composition(
      cute::Swizzle<1, 3, 3>{},
      cute::Layout<cute::Shape<cute::Shape<cute::_16,cute::_4>,
                              cute::Shape<cute::_16,cute::_2>>,
                   cute::Stride<cute::Stride<cute::_1,cute::_512>,
                                cute::Stride<cute::_16,cute::_256>>>{}));
  CUTE_HOST_DEVICE static constexpr unsigned cube(unsigned row, unsigned col) {
    return (row / 16) * (ValueTile / 16) + col / 16;
  }
  CUTE_HOST_DEVICE static constexpr unsigned offset(unsigned row, unsigned col) {
    return Layout{}(row,col);
  }
  CUTE_HOST_DEVICE static constexpr unsigned producer_base(unsigned warp, unsigned lane) {
    return ((warp / 2) * 4 + warp % 2) * 256 + (lane % 4) * 16 + lane / 4;
  }
  CUTE_HOST_DEVICE static constexpr unsigned producer_offset(unsigned base, unsigned fragment, unsigned slot) {
    return base + fragment * 512 + (slot % 4) * 64 + ((slot / 4) ^ (slot % 2)) * 8;
  }
  CUTE_DEVICE static void load(BF16 const* shared, unsigned row, unsigned col,
                               uint32_t (&fragment)[4]) {
    // Physical non-transpose delivers B(value,time) directly to native MMA.
    Physical::load(shared + cube(row,col) * 256, 0, 0, fragment);
  }
};
static_assert(ValueTile == 32 && Plan::Threads == 128 && sizeof(Storage) == 45568);
}  // namespace gdn_qsa::wy::residual_blayout
