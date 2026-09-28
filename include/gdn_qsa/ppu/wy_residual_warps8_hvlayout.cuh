#pragma once

#include "gdn_qsa/ppu/wy_residual_warps8_hlayout.cuh"

namespace gdn_qsa::wy::residual_warps8_hvlayout {
using residual_warps8_hlayout::StateTile;
using residual_warps8_hlayout::Plan;
using residual_warps8_hlayout::Storage;
using residual_warps8_hlayout::Key;
using residual_warps8_hlayout::Inverse;
using residual_warps8_hlayout::Value;
using residual_warps8_hlayout::BIntermediate;
using residual_warps8_hlayout::Snapshot;
using residual_warps8_hlayout::OutputSnapshot;

// Input V still uses Value's original AIU layout. After RESIDUAL_READY has
// retired every input-V reader, PR's BF16 results reuse the same buffer in
// B orientation. This layout has no additional copy or rounding operation.
struct PublishedValue : BIntermediate {
  // Offsets and pitches are BF16 ELEMENTS: private Vnew[group,value,time].
  // This is not the public input/output layout or the H snapshot pitch.
  CUTE_HOST_DEVICE static constexpr int64_t workspace_offset(int64_t group, unsigned t, unsigned v) {
    return tile_offset(group) + int64_t(v) * Chunk + t;
  }
  template <unsigned Threads>
  using Publication = StateVectorPlan<ValueTile, Chunk, 2, Threads>;
  template <unsigned Threads>
  CUTE_DEVICE static void publish(BF16 const* shared, BF16* global) {
    using Vectors = Publication<Threads>;
    unsigned const tid = unsigned(threadIdx.x);
    cute::for_each(cute::make_seq<Vectors::Iterations>{}, [&](auto iteration) {
      unsigned const i = Vectors::vector(tid, decltype(iteration)::value);
      unsigned const v = Vectors::row(i), t = Vectors::col(i);
      uint4 const packed = *reinterpret_cast<uint4 const*>(shared + offset(t,v));
      cutlass::arch::global_store<uint4,16>(packed, global + int64_t(v) * Chunk + t, true);
    });
  }
};

// Paired operand reader only. The final output matrix must keep its old
// row-oriented exchange/publisher. A square64x64 tile still needs one AIU
// cube, unlike the earlier H128x64 ->64x128 change.
struct OutputValue : aiu::Tile<OutputTile::Panel, Chunk> {
  using Physical = aiu::Tile<OutputTile::Panel, Chunk>;
  CUTE_DEVICE static void stage(BF16* shared, BF16 const* global) {
    Physical::stage(shared, global, Chunk, OutputTile::Panel);
  }
  CUTE_DEVICE static void load(BF16 const* shared, unsigned t, unsigned v, uint32_t (&fragment)[4]) {
    Physical::load(shared, v, t, fragment);
  }
};
static_assert(Chunk == 64 && Dim == 128 && ValueTile == 32 && OutputTile::Panel == 64);
static_assert(PublishedValue::Publication<Plan::Threads>::Iterations == 1);
static_assert(OutputValue::Physical::Cubes == 1 && sizeof(Storage) == 45568);
static_assert(PublishedValue::workspace_offset(1,7,11) == 8192 + 11*64 + 7);
}  // namespace gdn_qsa::wy::residual_warps8_hvlayout
