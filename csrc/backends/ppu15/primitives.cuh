#pragma once

#include <hggc_runtime.h>
#include <cutlass/bfloat16.h>
#include <cutlass/tfloat32.h>
#include <cute/arch/copy_ppu.hpp>
#include <cute/arch/copy_ppu0015_aiu.hpp>
#include "layout.cuh"

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ != 150
#error "actlizeLA ppu15 kernels require the native PPU1.5 target (arch150)"
#endif

namespace actlize_la::ppu15 {
using BF16 = cutlass::bfloat16_t;

template <int Rows, int Cols>
struct Tile : TileLayout<Rows, Cols> {
  using Layout = TileLayout<Rows, Cols>;
  using Writer = cute::PPU0015_AIU_LOAD<cute::Int<Rows * Layout::Width * 16>,
                                      BF16, false, Rows, Layout::Width>;
  CUTE_DEVICE static void stage(BF16* shared, BF16 const* global, int stride, int valid) {
    if (threadIdx.x != 0) return;
    cute::AiuDesc desc{};
    desc.dim_h = valid;
    desc.dim_w = Cols;
    desc.stride_w = stride;  // PPU1.5 has an explicit ELEMENT pitch.
    CUTE_UNROLL
    for (int cube = 0; cube < Layout::Cubes; ++cube)
      Writer::copy(shared + cube * Rows * Layout::Width, global, desc,
                   cube * Layout::Width, 0);
  }
  // A and B have different register orders on tc02. For a transposed
  // operand, row/col still name its ORIGINAL shared coordinates.
  template <bool OperandA, bool Transpose = false>
  CUTE_DEVICE static void load(BF16 const* shared, int row, int col, uint32_t (&r)[4]) {
    using Reader = cute::PPU0015_TSM_LD_SWZL<BF16, Rows, Layout::Width,
                                           !OperandA, Transpose, Layout::Cubes>;
    if constexpr (Transpose)
      Reader::copy(r, const_cast<BF16*>(shared), row, col % Layout::Width, col / Layout::Width);
    else
      Reader::copy(r, const_cast<BF16*>(shared), col % Layout::Width, row, col / Layout::Width);
  }
  template <int Threads>
  CUTE_DEVICE static void publish(BF16 const* shared, BF16* global, int64_t stride, int valid = Rows) {
    static_assert((Rows * Cols / 8) % Threads == 0);
    CUTE_UNROLL
    for (int it = 0; it < Rows * Cols / (8 * Threads); ++it) {
      int const i = (int(threadIdx.x) + it * Threads) * 8;
      int const row = i / Cols, col = i % Cols;
      if (row < valid)
        *reinterpret_cast<uint4*>(global + int64_t(row) * stride + col) =
            *reinterpret_cast<uint4 const*>(shared + Layout::offset(row, col));
    }
  }
};

template <class Atom>
CUTE_DEVICE void mma(float (&c)[8], uint32_t const (&a)[4], uint32_t const (&b)[4]) {
  Atom::fma(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],
            a[0],a[1],a[2],a[3],b[0],b[1],b[2],b[3],
            c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]);
}
CUTE_DEVICE void bf16_mma(float (&c)[8], uint32_t const (&a)[4], uint32_t const (&b)[4]) {
  mma<Bf16Atom>(c, a, b);
}

// Same three-product FP32 inverse as the retained PPU1.0 algorithm. Do not
// silently downgrade the inverse to BF16 or discard TF32 residual products.
CUTE_DEVICE void tf32_product(float (&acc)[8], float const* a, int lda,
                              float const* b, int ldb) {
  int const lane = int(threadIdx.x) % 32;
  CUTE_UNROLL
  for (int k = 0; k < 16; k += 8) {
    uint32_t ah[4], al[4], bh[4], bl[4];
    CUTE_UNROLL
    for (int s = 0; s < 4; ++s) {
      auto const ac = coordinate<typename cute::MMA_Traits<Tf32Atom>::ALayout, 16>(lane, s);
      auto const bc = coordinate<typename cute::MMA_Traits<Tf32Atom>::BLayout, 16>(lane, s);
      float const av = a[ac.row * lda + ac.col + k];
      float const bv = b[(bc.col + k) * ldb + bc.row];
      cutlass::tfloat32_t const at(av), bt(bv);
      ah[s] = at.raw(); bh[s] = bt.raw();
      al[s] = cutlass::tfloat32_t(av - float(at)).raw();
      bl[s] = cutlass::tfloat32_t(bv - float(bt)).raw();
    }
    mma<Tf32Atom>(acc, al, bh);
    mma<Tf32Atom>(acc, ah, bl);
    mma<Tf32Atom>(acc, ah, bh);
  }
}
CUTE_DEVICE void commit_wait() {
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();
}
}  // namespace actlize_la::ppu15
