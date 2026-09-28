#pragma once

#include "gdn_qsa/ppu/wy_residual_warps8_blayout.cuh"

namespace gdn_qsa::wy::residual_warps8_hlayout {
using residual_warps8_blayout::StateTile;
using residual_warps8_blayout::Plan;
using residual_warps8_blayout::Storage;
using residual_warps8_blayout::Key;
using residual_warps8_blayout::Inverse;
using residual_warps8_blayout::Value;
using residual_warps8_blayout::BIntermediate;

// H(k,v) has TWO consumers. Both see the same BF16 values: KH reads this
// B-oriented shared view; output reads private global snapshots in [v,k].
// Public FP32 initial/final H remains [k,v]. All pitches here are elements.
struct Snapshot {
  using Physical = aiu::Tile<16, 16>;
  using Layout = decltype(cute::composition(cute::Swizzle<1,3,3>{},
      cute::Layout<cute::Shape<cute::Shape<cute::_16,cute::_8>,
                              cute::Shape<cute::_16,cute::_2>>,
                   cute::Stride<cute::Stride<cute::_1,cute::_512>,
                                cute::Stride<cute::_16,cute::_256>>>{}));
  CUTE_HOST_DEVICE static constexpr unsigned cube(unsigned k, unsigned v) {
    return (k / 16) * (ValueTile / 16) + v / 16;
  }
  CUTE_HOST_DEVICE static constexpr unsigned offset(unsigned k, unsigned v) {
    return Layout{}(k,v);
  }
  CUTE_HOST_DEVICE static constexpr unsigned producer_base(unsigned warp, unsigned lane) {
    return cube(StateTile::k_row(warp,0), StateTile::column(warp)) * 256 +
           (lane % 4) * 16 + lane / 4;
  }
  CUTE_HOST_DEVICE static constexpr unsigned producer_offset(unsigned base, unsigned fragment, unsigned slot) {
    return base + fragment * 512 + (slot % 4) * 64 + ((slot / 4) ^ (slot % 2)) * 8;
  }
  CUTE_HOST_DEVICE static constexpr int64_t workspace_offset(int64_t group, unsigned k, unsigned v) {
    return state_offset(group) + int64_t(v) * Dim + k;
  }
  CUTE_DEVICE static void load(BF16 const* shared, unsigned k, unsigned v, uint32_t (&fragment)[4]) {
    Physical::load(shared + cube(k,v) * 256, 0, 0, fragment);
  }
  template <unsigned Threads>
  using Publication = StateVectorPlan<ValueTile, Dim, 2, Threads>;
  template <unsigned Threads>
  CUTE_DEVICE static void publish(BF16 const* shared, BF16* global, int64_t stride) {
    using Vectors = Publication<Threads>;
    unsigned const tid = unsigned(threadIdx.x);
    cute::for_each(cute::make_seq<Vectors::Iterations>{}, [&](auto iteration) {
      unsigned const i = Vectors::vector(tid, decltype(iteration)::value);
      unsigned const v = Vectors::row(i), k = Vectors::col(i);
      uint4 const packed = *reinterpret_cast<uint4 const*>(shared + offset(k,v));
      cutlass::arch::global_store<uint4,16>(packed, global + int64_t(v) * stride + k, true);
    });
  }
};

// Paired consumer of private H[v,k].64x128 physical tile uses TWO native
// 64x64 AIU cubes rather than the old one128x64 cube; bytes are unchanged,
// but its extra bulk-copy instruction is a real cost to measure.
struct OutputSnapshot : aiu::Tile<OutputTile::Panel, Dim> {
  using Physical = aiu::Tile<OutputTile::Panel, Dim>;
  CUTE_DEVICE static void stage(BF16* shared, BF16 const* global) {
    Physical::stage(shared, global, Dim, OutputTile::Panel);
  }
  CUTE_DEVICE static void load(BF16 const* shared, unsigned k, unsigned v, uint32_t (&fragment)[4]) {
    Physical::load(shared, v, k, fragment);
  }
};
static_assert(Dim == 128 && ValueTile == 32 && OutputTile::Panel == 64);
static_assert(sizeof(Storage) == 45568 && Plan::Threads == 256);
static_assert(Snapshot::workspace_offset(1,7,11) == 16384 + 11*128 + 7);
}  // namespace gdn_qsa::wy::residual_warps8_hlayout
