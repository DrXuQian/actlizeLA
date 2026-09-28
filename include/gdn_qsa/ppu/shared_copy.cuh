#pragma once

#include <cute/arch/copy_ppu0010_aiu.hpp>
#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/tensor.hpp>
#include "gdn_qsa/ppu/fragment.cuh"

namespace gdn_qsa::ppu {

// Independent 16x16 physical cubes. Within a cube the AIU byte swizzle is
// bit3 ^= bit6 in HALF-ELEMENT offsets. Eight adjacent half elements remain
// contiguous, preserving every original 16-byte cp.async/vector transaction.
template <int Rows, int Columns>
using RowLayout = decltype(cute::composition(
    cute::Swizzle<1, 3, 3>{},
    cute::Layout<
        cute::Shape<cute::Shape<cute::_16, cute::Int<Rows / 16>>,
                    cute::Shape<cute::_16, cute::Int<Columns / 16>>>,
        cute::Stride<cute::Stride<cute::_16, cute::_256>,
                     cute::Stride<cute::_1, cute::Int<Rows * 16>>>>{}));

template <int Rows, int Columns>
using ColumnLayout = decltype(cute::composition(
    RowLayout<Columns, Rows>{},
    cute::make_layout(cute::make_shape(cute::Int<Rows>{}, cute::Int<Columns>{}),
                      cute::make_stride(cute::Int<Columns>{}, cute::_1{}))));

enum class CopyRole { A, AT, B, C, CT, StoreC };

template <CopyRole Role, class Tensor>
struct SharedTile { Tensor tensor; int lane; };

template <CopyRole Role>
struct SharedCopyThread {
  int lane;
  template <class T> CUTE_HOST_DEVICE auto partition_S(T const& t) const {
    return SharedTile<Role, T>{t, lane};
  }
  template <class T> CUTE_HOST_DEVICE auto partition_D(T const& t) const {
    return SharedTile<Role, T>{t, lane};
  }
  // Match CuTe's retile contract: return a NON-OWNING tensor by value.
  // Returning T& is not equivalent: `auto view = retile_D(fragment)` copies
  // an owning ArrayEngine, and the subsequent load then misses the operand.
  template <class T> CUTE_HOST_DEVICE auto retile_D(T& t) const {
    return cute::make_tensor(t.data(), t.layout());
  }
  template <class T> CUTE_HOST_DEVICE auto retile_S(T const& t) const {
    return cute::make_tensor(t.data(), t.layout());
  }
};

template <CopyRole Role>
struct SharedCopy {
  CUTE_HOST_DEVICE auto get_thread_slice(int lane) const {
    return SharedCopyThread<Role>{lane};
  }
  CUTE_HOST_DEVICE auto get_slice(int lane) const { return get_thread_slice(lane); }
};

// The incoming tensor is the original local_tile(view,16x16,coordinate), so
// its (0,0) address includes the absolute tile offset and transpose view.
// A/AT/B delivery is one hardware matrix load per 16x16 fragment.
template <CopyRole Role, class T, class D>
CUTE_DEVICE void copy(SharedCopy<Role>, SharedTile<Role, T> const& source,
                      D&& destination) {
  static_assert(cute::size(typename std::remove_reference<D>::type{}) == 8);
  using Element = typename T::value_type;
  auto* base = const_cast<Element*>(&source.tensor(0, 0));
  uint32_t loaded[4];
  constexpr bool trans = Role == CopyRole::AT || Role == CopyRole::CT;
  cute::PPU0010_TSM_LD_SWZL<Element, 16, 16, true, trans>::copy(
      loaded, base, 0, 0);
  auto* target = reinterpret_cast<uint32_t*>(&destination(0));
  if constexpr (Role == CopyRole::C || Role == CopyRole::CT) {
    remap<FragmentMap::Operand, FragmentMap::Result, false>(loaded, target);
  } else {
    CUTE_UNROLL
    for (int word = 0; word < 4; ++word) target[word] = loaded[word];
  }
}

template <CopyRole Role, class S, class T>
CUTE_DEVICE void copy(SharedCopy<Role>, S const& source,
                      SharedTile<Role, T> const& destination) {
  static_assert(cute::size(S{}) == 8);
  CUTE_UNROLL
  for (int slot = 0; slot < 8; ++slot) {
    auto const c = result_coord(destination.lane, slot);
    destination.tensor(c.row, c.col) = source(slot);
  }
}

}  // namespace gdn_qsa::ppu
