#pragma once

#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/atom/mma_traits_ppu0015.hpp>

namespace actlize_la::ppu15 {
struct Coordinate { int row, col; };
using Bf16Atom = cute::PPU0015_16x16x16_F32BF16BF16F32_TN;
using Tf32Atom = cute::PPU0015_16x16x8_F32TF32TF32F32_TN;

// Use the pinned native traits as the register ABI, not PPU1.0's coordinates.
template <class Layout, int Rows>
CUTE_HOST_DEVICE constexpr Coordinate coordinate(int lane, int slot) {
  int const i = Layout{}(cute::make_coord(lane, slot));
  return {i % Rows, i / Rows};
}
CUTE_HOST_DEVICE constexpr Coordinate result_coord(int lane, int slot) {
  return coordinate<typename cute::MMA_Traits<Bf16Atom>::CLayout, 16>(lane, slot);
}
CUTE_HOST_DEVICE constexpr int row_half(int slot) { return (slot / 2) % 2; }

// Paired with PPU0015_AIU_LOAD + PPU0015_TSM_LD_SWZL. BF16 cubes are
// 128B (mode0) or 64B (mode1) wide. Offsets are BF16 ELEMENTS, never bytes.
// The layout is the native FA-sail PPU1.5 SW128/SW64 layout, NOT the tc01
// rotating 16x16 cube. Cube origins must be 128B aligned.
template <int Rows, int Cols>
struct TileLayout {
  static_assert(Rows % 16 == 0 && Cols % 32 == 0);
  static constexpr int Width = Cols >= 64 ? 64 : 32;
  static constexpr int Cubes = Cols / Width;
  using Swizzle = cute::Swizzle<Width == 64 ? 3 : 2, 3, 3>;
  CUTE_HOST_DEVICE static constexpr int offset(int row, int col) {
    return (col / Width) * Rows * Width + Swizzle{}(row * Width + col % Width);
  }
};

struct StateTile {
  static constexpr int Threads = 256, KFragments = 2, ValueFragments = 1;
  CUTE_HOST_DEVICE static constexpr int k_row(int warp, int f) { return (warp / 2) * 32 + f * 16; }
  CUTE_HOST_DEVICE static constexpr int value_row(int warp) { return (warp / 2) * 16; }
  CUTE_HOST_DEVICE static constexpr int col(int warp) { return (warp % 2) * 16; }
};
struct OutputTile {
  static constexpr int Threads = 256, Panel = 64, Fragments = 2;
  CUTE_HOST_DEVICE static constexpr int row(int warp) { return (warp / 2) * 16; }
  CUTE_HOST_DEVICE static constexpr int col(int warp, int f) { return (warp % 2) * 32 + f * 16; }
};
}  // namespace actlize_la::ppu15
