#pragma once

#include "gdn_qsa/ppu/wy_mma.cuh"

namespace gdn_qsa::wy::inverse_register {

// C(k,n) -> B(n,k) for one K8 half of the native TF32 atom. Source values
// stay FP32; this changes delivery only, never the high/residual decomposition.
CUTE_HOST_DEVICE constexpr gdn_qsa::ppu::Owner b_owner(unsigned lane, int word, int half) {
  auto const b = tf32_coord(int(lane), word);
  return gdn_qsa::ppu::result_owner(b.col + 8 * half, b.row);
}

template <int Half, int Word, class Exchange>
CUTE_HOST_DEVICE float b_value(unsigned lane, Exchange const& exchange) {
  static_assert(Half >= 0 && Half < 2 && Word >= 0 && Word < 4);
  constexpr int lo = b_owner(0, Word, Half).slot;
  auto const owner = b_owner(lane, Word, Half);
  // Every lane must exchange both candidates before selecting. Selecting a
  // source slot first would use the source lane's high bit, not the receiver's.
  float const low = exchange.template get<lo>(owner.lane);
  float const high = exchange.template get<lo + 1>(owner.lane);
  return (lane & 16u) ? high : low;
}

struct WarpExchange {
  float const (&values)[8];
  template <int Slot>
  CUTE_DEVICE float get(unsigned peer) const {
#if defined(__HGGC_ARCH__)
    return __shfl_sync(0xffffffffu, values[Slot], peer);
#else
    CUTE_INVALID_CONTROL_PATH("inverse register delivery requires a PPU warp");
    return 0.0f;
#endif
  }
};

// Same two K8 slices and ordered three-product TF32 arithmetic as tf32_product.
// The left diagonal still comes from its admitted shared-memory publication.
CUTE_DEVICE void product(float (&acc)[8], float const* a, int lda,
                         float const (&b)[8]) {
  unsigned const lane = unsigned(threadIdx.x) & 31u;
  WarpExchange const exchange{b};
  cute::for_each(cute::make_seq<2>{}, [&](auto half) {
    constexpr int Half = decltype(half)::value;
    uint32_t ah[4], al[4], bh[4], bl[4];
    cute::for_each(cute::make_seq<4>{}, [&](auto word) {
      constexpr int Word = decltype(word)::value;
      auto const rc = tf32_coord(int(lane), Word);
      float const av = a[rc.row * lda + rc.col + 8 * Half];
      float const bv = b_value<Half, Word>(lane, exchange);
      cutlass::tfloat32_t const at(av), bt(bv);
      ah[Word] = at.raw(); bh[Word] = bt.raw();
      al[Word] = cutlass::tfloat32_t(av - float(at)).raw();
      bl[Word] = cutlass::tfloat32_t(bv - float(bt)).raw();
    });
    using Op = cute::PPU0010_16x16x8_F32TF32TF32F32_TN;
    mma<Op>(acc, al, bh);
    mma<Op>(acc, ah, bl);
    mma<Op>(acc, ah, bh);
  });
}
}  // namespace gdn_qsa::wy::inverse_register
