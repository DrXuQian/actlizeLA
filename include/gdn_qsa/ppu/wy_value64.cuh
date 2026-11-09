#pragma once
#include "gdn_qsa/ppu/wy_gate_coefficients.cuh"
#include <cstddef>

namespace gdn_qsa::wy::value64 {
constexpr unsigned Width = 64;
template <unsigned TileWidth, unsigned ThreadCount = TileWidth * 8>
struct GeometryT {
  static_assert(TileWidth == 16 || TileWidth == 32 || TileWidth == 64);
  static_assert(ThreadCount == TileWidth * 8 || (TileWidth == 64 && ThreadCount == 256));
  static constexpr unsigned WarpColumns = TileWidth / 16;
  static constexpr unsigned Threads = ThreadCount;
  static constexpr unsigned KFragments = Dim * TileWidth / (Threads * 8);
  static constexpr unsigned ValueFragments = Chunk * TileWidth / (Threads * 8);
  CUTE_HOST_DEVICE static constexpr unsigned k_row(unsigned w, unsigned f) { return (w / WarpColumns) * (16 * KFragments) + f * 16; }
  CUTE_HOST_DEVICE static constexpr unsigned value_row(unsigned w, unsigned f) { return (w / WarpColumns) * (16 * ValueFragments) + f * 16; }
  CUTE_HOST_DEVICE static constexpr unsigned column(unsigned w) { return (w % WarpColumns) * 16; }
};
using Geometry = GeometryT<Width>;
using Key = aiu::Tile<Chunk, Dim>;
using Inverse = aiu::Tile<Chunk, Chunk>;
using Value = aiu::Tile<Chunk, Width>;

// 微cube内C→B排列不变；逻辑宽度决定物理行距，线程数只决定所有权。
template <unsigned Rows, unsigned TileWidth = Width, unsigned ThreadCount = TileWidth * 8>
struct MicroB {
  using Physical = aiu::Tile<16, 16>;
  using Layout = decltype(cute::composition(cute::Swizzle<1, 3, 3>{},
      cute::Layout<cute::Shape<cute::Shape<cute::_16, cute::Int<Rows / 16>>,
                              cute::Shape<cute::_16, cute::Int<TileWidth / 16>>>,
                   cute::Stride<cute::Stride<cute::_1, cute::Int<TileWidth * 16>>,
                                cute::Stride<cute::_16, cute::_256>>>{}));
  CUTE_HOST_DEVICE static constexpr unsigned cube(unsigned row, unsigned col) { return (row / 16) * (TileWidth / 16) + col / 16; }
  CUTE_HOST_DEVICE static constexpr unsigned offset(unsigned row, unsigned col) { return Layout{}(row, col); }
  CUTE_HOST_DEVICE static constexpr unsigned producer_base(unsigned warp, unsigned lane) {
    unsigned const row = Rows == Dim ? GeometryT<TileWidth, ThreadCount>::k_row(warp, 0) : GeometryT<TileWidth, ThreadCount>::value_row(warp, 0);
    return cube(row, GeometryT<TileWidth, ThreadCount>::column(warp)) * 256 + (lane % 4) * 16 + lane / 4;
  }
  CUTE_HOST_DEVICE static constexpr unsigned producer_offset(unsigned base, unsigned fragment, unsigned slot) {
    return base + fragment * (TileWidth * 16) + (slot % 4) * 64 + ((slot / 4) ^ (slot % 2)) * 8;
  }
  CUTE_DEVICE static void load(BF16 const* shared, unsigned row, unsigned col, uint32_t (&fragment)[4]) {
    Physical::load(shared + cube(row, col) * 256, 0, 0, fragment);
  }
};
template <unsigned TileWidth, unsigned ThreadCount = TileWidth * 8>
struct SnapshotT : MicroB<Dim, TileWidth, ThreadCount> {
  using MicroB<Dim, TileWidth, ThreadCount>::offset;
  CUTE_HOST_DEVICE static constexpr int64_t workspace_offset(int64_t group, unsigned k, unsigned v) {
    return state_offset(group) + int64_t(v) * Dim + k;
  }
  CUTE_DEVICE static void publish(BF16 const* shared, BF16* global) {
    using P = StateVectorPlan<TileWidth, Dim, 2, GeometryT<TileWidth, ThreadCount>::Threads>;
    unsigned const tid = unsigned(threadIdx.x);
    CUTE_UNROLL
    for (unsigned it = 0; it < P::Iterations; ++it) {
      unsigned const i = P::vector(tid, it), v = P::row(i), k = P::col(i);
      uint4 const packed = *reinterpret_cast<uint4 const*>(shared + offset(k, v));
      cutlass::arch::global_store<uint4, 16>(packed, global + int64_t(v) * Dim + k, true);
    }
  }
};
template <unsigned TileWidth, unsigned ThreadCount = TileWidth * 8>
struct PublishedValueT : MicroB<Chunk, TileWidth, ThreadCount> {
  using MicroB<Chunk, TileWidth, ThreadCount>::offset;
  CUTE_HOST_DEVICE static constexpr int64_t workspace_offset(int64_t group, unsigned t, unsigned v) {
    return tile_offset(group) + int64_t(v) * Chunk + t;
  }
  CUTE_DEVICE static void publish(BF16 const* shared, BF16* global) {
    using P = StateVectorPlan<TileWidth, Chunk, 2, GeometryT<TileWidth, ThreadCount>::Threads>;
    unsigned const tid = unsigned(threadIdx.x);
    CUTE_UNROLL
    for (unsigned it = 0; it < P::Iterations; ++it) {
      unsigned const i = P::vector(tid, it), v = P::row(i), t = P::col(i);
      uint4 const packed = *reinterpret_cast<uint4 const*>(shared + offset(t, v));
      cutlass::arch::global_store<uint4, 16>(packed, global + int64_t(v) * Chunk + t, true);
    }
  }
};
template <unsigned TileWidth>
struct IntermediateT {
  alignas(128) BF16 snapshot[Dim * TileWidth];
  alignas(128) BF16 value[Chunk * TileWidth];
  alignas(128) BF16 residual[Chunk * TileWidth];
};
template <unsigned TileWidth>
struct StorageT {
  alignas(128) BF16 k[Chunk * Dim], inverse[Chunk * Chunk];
  union {
    alignas(128) IntermediateT<TileWidth> math;
    alignas(128) float final_h[Dim * TileWidth];
  };
  alignas(128) BF16 scaled[Chunk * TileWidth];
  float beta[Chunk], gates[Chunk];
  gate_cache::Coefficients coefficients;
};
static_assert(sizeof(IntermediateT<64>) == sizeof(float) * Dim * 64);
static_assert(sizeof(StorageT<64>) == 66560);
static_assert(offsetof(StorageT<64>, final_h) == offsetof(StorageT<64>, math));
static_assert(GeometryT<64, 256>::KFragments == 4 && GeometryT<64, 256>::ValueFragments == 2);
}  // namespace gdn_qsa::wy::value64
