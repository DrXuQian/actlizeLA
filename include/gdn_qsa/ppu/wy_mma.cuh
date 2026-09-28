#pragma once

#include "gdn_qsa/ppu/shared_copy.cuh"
#include <cutlass/tfloat32.h>

namespace gdn_qsa::wy {
using BF16 = cutlass::bfloat16_t;
using gdn_qsa::ppu::result_coord;

CUTE_HOST_DEVICE constexpr gdn_qsa::ppu::Owner state_b_owner(unsigned lane, int slot) {
  return {int((2 * (lane & 3u) + slot % 2) * 4 + ((lane >> 2) & 3u)),
          ((slot / 2) % 2) * 4 + (slot / 4) * 2 + int((lane & 16u) != 0)};
}
CUTE_HOST_DEVICE constexpr gdn_qsa::ppu::Coordinate tf32_coord(int lane, int slot) {
  return {lane / 4 + (slot / 2) * 8, lane % 4 + (slot % 2) * 4};
}

template <int Rows, int Cols>
CUTE_HOST_DEVICE constexpr int swizzle(int row, int col) {
  return gdn_qsa::ppu::RowLayout<Rows, Cols>{}(row, col);
}

template <int Rows, int Cols, bool Transpose = false>
CUTE_DEVICE void load(BF16 const* p, int row, int col, uint32_t (&r)[4]) {
  cute::PPU0010_TSM_LD_SWZL<BF16, 16, 16, true, Transpose>::copy(
      r, const_cast<BF16*>(p + swizzle<Rows, Cols>(row, col)), 0, 0);
}

template <class Atom>
CUTE_DEVICE void mma(float (&c)[8], uint32_t const (&a)[4], uint32_t const (&b)[4]) {
  Atom::fma(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
            a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
}
CUTE_DEVICE void bf16_mma(float (&c)[8], uint32_t const (&a)[4], uint32_t const (&b)[4]) {
  mma<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>(c, a, b);
}

CUTE_DEVICE void state_to_b(float const (&h)[8], uint32_t (&b)[4]) {
#if defined(__HGGC_ARCH__)
  // C(row=K,col=V) -> B(row=V,col=K), specialized to the native 16x16
  // traits. Every destination uses two possible source slots; exchange both
  // BEFORE selecting with the destination lane's high bit.
  unsigned const lane = unsigned(threadIdx.x) & 31u;
  cute::for_each(cute::make_seq<4>{}, [&](auto word) {
    uint32_t packed = 0;
    cute::for_each(cute::make_seq<2>{}, [&](auto half) {
      constexpr int slot = decltype(word)::value * 2 + decltype(half)::value;
      constexpr int source0 = state_b_owner(0, slot).slot;
      auto const peer = state_b_owner(lane, slot);
      float const a = __shfl_sync(0xffffffffu, h[source0], peer.lane);
      float const c = __shfl_sync(0xffffffffu, h[source0 + 1], peer.lane);
      packed |= uint32_t(BF16((lane & 16u) ? c : a).raw()) << (16 * decltype(half)::value);
    });
    b[decltype(word)::value] = packed;
  });
#else
  CUTE_INVALID_CONTROL_PATH("PPU register conversion requires device execution");
#endif
}

// FP32 block inverse merges use TF32 high/residual products, not BF16
// truncation of the solve. The ignored residual*residual term is explicit.
CUTE_DEVICE void tf32_product(float (&acc)[8], float const* a, int lda,
                              float const* b, int ldb) {
  int const lane = int(threadIdx.x) & 31;
  CUTE_UNROLL
  for (int k = 0; k < 16; k += 8) {
    uint32_t ah[4], al[4], bh[4], bl[4];
    CUTE_UNROLL
    for (int s = 0; s < 4; ++s) {
      auto const rc = tf32_coord(lane, s);
      int const r = rc.row, c = rc.col + k;
      float const av = a[r * lda + c], bv = b[c * ldb + r];
      cutlass::tfloat32_t const at(av), bt(bv);
      ah[s] = at.raw(); bh[s] = bt.raw();
      al[s] = cutlass::tfloat32_t(av - float(at)).raw();
      bl[s] = cutlass::tfloat32_t(bv - float(bt)).raw();
    }
    using Op = cute::PPU0010_16x16x8_F32TF32TF32F32_TN;
    mma<Op>(acc, al, bh);
    mma<Op>(acc, ah, bl);
    mma<Op>(acc, ah, bh);
  }
}
}  // namespace gdn_qsa::wy
