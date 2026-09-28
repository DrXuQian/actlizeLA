#pragma once

#include "gdn_qsa/ppu/wy_mma.cuh"
#include "gdn_qsa/ppu/wy_state_address.cuh"
#include <cutlass/arch/memory.h>
#include <climits>

namespace gdn_qsa::wy::aiu {

// One contract for the bulk writer, scalar intermediate stores and matrix
// reader. All offsets/strides below are BF16 ELEMENTS, never bytes.
// A cube spans the full tile height and at most128 contiguous bytes.
template <unsigned Rows, unsigned Cols>
struct Tile {
  static_assert(Rows % 16 == 0 && Cols % 16 == 0);
  static constexpr unsigned CubeWidth = Cols < 64 ? Cols : 64;
  static constexpr unsigned Cubes = Cols / CubeWidth;
  static_assert(Cols % CubeWidth == 0);
  using Writer = cute::PPU0010_AIU_LOAD<
      cute::Int<Rows * CubeWidth * 16>, BF16, false, true>;
  template <bool Transpose>
  using Reader = cute::PPU0010_TSM_LD_SWZL<
      BF16, Rows, CubeWidth, true, Transpose, Cubes>;

  // This rotation is NOT just an XOR for slices2/3. Validate against the
  // actual actlize ppu_tsm_ld_swzl_sim, not a second copy of this expression.
  CUTE_HOST_DEVICE static constexpr unsigned offset(unsigned row, unsigned col) {
    unsigned const cube = col / CubeWidth, within = col % CubeWidth;
    unsigned const slice = within / 16, line = row / 4;
    unsigned const rotation = ((slice & 1u) << 2) | (slice & 2u);
    unsigned const vector = (((row % 4) * 2 + (within % 16) / 8) ^ (line % 2));
    return cube * Rows * CubeWidth + slice * Rows * 16 + line * 64 +
           ((vector + rotation) % 8) * 8 + within % 8;
  }

  CUTE_HOST_DEVICE static constexpr bool admitted_stride(int64_t stride) {
    return stride >= Cols && stride <= INT_MAX;
  }
  CUTE_HOST_DEVICE static cute::AiuDesc descriptor(int stride, int valid_rows) {
    cute::AiuDesc desc{};
    desc.dim_h = valid_rows;  // padz excludes subsequent chunks/batches
    desc.dim_w = stride;     // row pitch, including GVA/interleaved heads
    desc.cube_h = Rows;
    desc.cube_w = CubeWidth;
    return desc;
  }

  CUTE_DEVICE static void stage(BF16* shared, BF16 const* global,
                                int stride, int valid_rows) {
    // AIU is a single-issuer bulk operation, not one scalar copy per lane.
    // Commit/wait and the CTA handoff remain at the caller's lifetime seam.
    if (threadIdx.x == 0) {
      auto const desc = descriptor(stride, valid_rows);
      CUTE_UNROLL
      for (unsigned cube = 0; cube < Cubes; ++cube)
        Writer::copy(shared + cube * Rows * CubeWidth, global, desc,
                     cube * CubeWidth, 0);
    }
  }

  template <bool Transpose = false>
  CUTE_DEVICE static void load(BF16 const* shared, unsigned row, unsigned col,
                               uint32_t (&fragment)[4]) {
    // Pass the cube plus its logical coordinates, not a software-swizzled
    // pointer to every16x16 microtile. Both endpoints share this geometry.
    if constexpr (Transpose)
      Reader<true>::copy(fragment, const_cast<BF16*>(shared), row,
                         col % CubeWidth, col / CubeWidth);
    else
      Reader<false>::copy(fragment, const_cast<BF16*>(shared), col % CubeWidth,
                          row, col / CubeWidth);
  }

  template <unsigned Threads>
  CUTE_DEVICE static void publish(BF16 const* shared, BF16* global,
                                  int64_t stride, unsigned valid_rows = Rows) {
    using Plan = StateVectorPlan<Rows, Cols, 2, Threads>;
    unsigned const tid = unsigned(threadIdx.x);
    cute::for_each(cute::make_seq<Plan::Iterations>{}, [&](auto iteration) {
      unsigned const i = Plan::vector(tid, decltype(iteration)::value);
      unsigned const row = Plan::row(i), col = Plan::col(i);
      if (row < valid_rows) {
        uint4 const packed = *reinterpret_cast<uint4 const*>(shared + offset(row, col));
        cutlass::arch::global_store<uint4, 16>(packed, global + int64_t(row) * stride + col, true);
      }
    });
  }
};

}  // namespace gdn_qsa::wy::aiu
