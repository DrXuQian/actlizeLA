#pragma once

#include "gdn_qsa/ppu/wy_mma.cuh"
#include <cutlass/arch/memory.h>

namespace gdn_qsa::wy {

// Ownership of aligned 16-byte transactions. Keep this shared with the host
// gate; global strides/layouts do not change when the writer is vectorized.
template <int Rows, int Cols, int ElementBytes>
struct VectorTile {
  static constexpr int Width = 16 / ElementBytes;
  static_assert(Cols % Width == 0);
  static constexpr int Count = Rows * Cols / Width;
  CUTE_HOST_DEVICE static constexpr int row(int vector) { return vector / (Cols / Width); }
  CUTE_HOST_DEVICE static constexpr int col(int vector) { return vector % (Cols / Width) * Width; }
};

template <int Rows, int Cols, int Threads>
CUTE_DEVICE void publish_bf16(BF16 const* shared, BF16* global, int64_t stride,
                              int tid, int valid_rows = Rows) {
  using Tile = VectorTile<Rows, Cols, 2>;
  for (int i = tid; i < Tile::Count; i += Threads) {
    int const r = Tile::row(i), c = Tile::col(i);
    if (r < valid_rows) {
      uint4 const packed = *reinterpret_cast<uint4 const*>(shared + swizzle<Rows, Cols>(r, c));
      cutlass::arch::global_store<uint4, 16>(packed, global + r * stride + c, true);
    }
  }
}

template <int Rows, int Cols, int Threads>
CUTE_DEVICE void publish_fp32(float const* shared, float* global, int64_t stride, int tid) {
  using Tile = VectorTile<Rows, Cols, 4>;
  for (int i = tid; i < Tile::Count; i += Threads) {
    int const r = Tile::row(i), c = Tile::col(i);
    uint4 const packed = *reinterpret_cast<uint4 const*>(shared + r * Cols + c);
    cutlass::arch::global_store<uint4, 16>(packed, global + r * stride + c, true);
  }
}

// Warp-private 16x16 exchange. No inter-warp barrier or arithmetic/rounding
// change: each native FP32 accumulator is converted exactly once to BF16.
CUTE_DEVICE void stage_fragment(float const (&values)[8], BF16* shared, int lane) {
  CUTE_UNROLL
  for (int s = 0; s < 8; ++s) {
    auto const rc = result_coord(lane, s);
    shared[swizzle<16, 16>(rc.row, rc.col)] = BF16(values[s]);
  }
}

}  // namespace gdn_qsa::wy
