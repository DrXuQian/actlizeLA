#pragma once

#include "gdn_qsa/ppu/wy_gate_coefficients.cuh"

namespace gdn_qsa::wy::geometry {

// Geometry changes ownership, not reduction order or intermediate precision.
// Two N warps for V32; one for V16. Each warp owns whole native m16n16
// output fragments; no split-K, idle warp or cross-warp accumulator reduction.
template<int Columns, int Warps>
struct Config {
  static_assert((Columns == 32 && (Warps == 4 || Warps == 8)) ||
                (Columns == 16 && Warps == 4), "unproved state geometry");
  static constexpr int ValueTile = Columns;
  struct StateTile {
    static constexpr int Threads = Warps * 32;
    static constexpr int NWarps = Columns / 16;
    static constexpr int RowWarps = Warps / NWarps;
    static constexpr int KFragments = (Dim / 16) / RowWarps;
    static constexpr int ValueFragments = (Chunk / 16) / RowWarps;
    CUTE_HOST_DEVICE static constexpr int k_row(int warp, int fragment) {
      return (warp / NWarps) * (KFragments * 16) + fragment * 16;
    }
    CUTE_HOST_DEVICE static constexpr int value_row(int warp, int fragment) {
      return (warp / NWarps) * (ValueFragments * 16) + fragment * 16;
    }
    CUTE_HOST_DEVICE static constexpr int column(int warp) { return (warp % NWarps) * 16; }
  };
  struct Plan : residual::Plan { static constexpr int Threads = StateTile::Threads; };
  using Key = aiu::Tile<Chunk, Dim>;
  using Inverse = aiu::Tile<Chunk, Chunk>;
  using Value = aiu::Tile<Chunk, Columns>;

  // Native B-oriented 16x16 microcubes. All offsets are BF16 ELEMENTS.
  // Physical map is independent of warp count; production C owners and
  // native ld.swzl B readers are exhaustively checked in L040.
  template<int Rows, bool SnapshotPlane>
  struct BPlane {
    using Physical = aiu::Tile<16, 16>;
    using Layout = decltype(cute::composition(cute::Swizzle<1,3,3>{},
        cute::Layout<cute::Shape<cute::Shape<cute::_16,cute::Int<Rows/16>>,
                                cute::Shape<cute::_16,cute::Int<Columns/16>>>,
                     cute::Stride<cute::Stride<cute::_1,cute::Int<Columns*16>>,
                                  cute::Stride<cute::_16,cute::_256>>>{}));
    CUTE_HOST_DEVICE static constexpr unsigned cube(unsigned row, unsigned col) {
      return (row / 16) * (Columns / 16) + col / 16;
    }
    CUTE_HOST_DEVICE static constexpr unsigned offset(unsigned row, unsigned col) {
      return Layout{}(row, col);
    }
    CUTE_HOST_DEVICE static constexpr unsigned producer_base(unsigned warp, unsigned lane) {
      unsigned const row = SnapshotPlane ? StateTile::k_row(warp, 0) : StateTile::value_row(warp, 0);
      return cube(row, StateTile::column(warp)) * 256 + (lane % 4) * 16 + lane / 4;
    }
    CUTE_HOST_DEVICE static constexpr unsigned producer_offset(unsigned base, unsigned fragment, unsigned slot) {
      return base + fragment * (Columns * 16) + (slot % 4) * 64 + ((slot / 4) ^ (slot % 2)) * 8;
    }
    CUTE_HOST_DEVICE static constexpr int64_t workspace_offset(int64_t group, unsigned row, unsigned col) {
      return group * int64_t(Rows) * Dim + int64_t(col) * Rows + row;
    }
    CUTE_DEVICE static void load(BF16 const* shared, unsigned row, unsigned col, uint32_t (&fragment)[4]) {
      Physical::load(shared + cube(row,col) * 256, 0, 0, fragment);
    }
    template<unsigned Threads>
    using Publication = StateVectorPlan<Columns, Rows, 2, Threads>;
    template<unsigned Threads>
    CUTE_DEVICE static void publish(BF16 const* shared, BF16* global, int64_t stride = Rows) {
      using Vectors = Publication<Threads>;
      unsigned const tid = unsigned(threadIdx.x);
      cute::for_each(cute::make_seq<Vectors::Iterations>{}, [&](auto iteration) {
        unsigned const i = Vectors::vector(tid, decltype(iteration)::value);
        unsigned const col = Vectors::row(i), row = Vectors::col(i);
        uint4 const packed = *reinterpret_cast<uint4 const*>(shared + offset(row,col));
        cutlass::arch::global_store<uint4,16>(packed, global + int64_t(col) * stride + row, true);
      });
    }
  };
  using Snapshot = BPlane<Dim, true>;
  using BIntermediate = BPlane<Chunk, false>;
  using PublishedValue = BIntermediate;

  struct Matrices {
    union {
      alignas(128) BF16 k[Chunk * Dim];
      alignas(128) float final_h[Dim * Columns];
    };
    alignas(128) BF16 inverse[Chunk * Chunk];
    alignas(128) BF16 snapshot[Dim * Columns];
    alignas(128) BF16 value[Chunk * Columns];
    alignas(128) BF16 residual[Chunk * Columns];
    alignas(128) BF16 scaled[Chunk * Columns];
    float gates[Chunk], beta[Chunk];
  };
  struct Storage {
    Matrices matrices;
    gate_cache::Coefficients coefficients;
  };
  static_assert(Warps * StateTile::KFragments * 256 == Dim * Columns);
  static_assert(Warps * StateTile::ValueFragments * 256 == Chunk * Columns);
  static_assert(sizeof(Storage) == (Columns == 32 ? 46080 : 35840));
};
}  // namespace gdn_qsa::wy::geometry
