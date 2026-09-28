#pragma once

#include <cute/atom/mma_traits_ppu0010.hpp>
#include <cute/tensor.hpp>

namespace gdn_qsa::ppu {

struct Coordinate { int row; int col; };
struct Owner { int lane; int slot; };

// Native PPU0010 fragment ABI. All conversion paths below share these maps;
// the host gate independently compares them against the real MMA traits.
CUTE_HOST_DEVICE constexpr Coordinate operand_coord(int lane, int slot) {
  return {lane / 4 + (slot / 4) * 8,
          (lane % 4) * 2 + slot % 2 + ((slot / 2) % 2) * 8};
}
CUTE_HOST_DEVICE constexpr Coordinate result_coord(int lane, int slot) {
  return {lane / 4 + (slot / 4) * 8, lane % 4 + (slot % 4) * 4};
}
CUTE_HOST_DEVICE constexpr Owner operand_owner(int row, int col) {
  return {(row % 8) * 4 + (col % 8) / 2,
          (row / 8) * 4 + (col / 8) * 2 + col % 2};
}
CUTE_HOST_DEVICE constexpr Owner result_owner(int row, int col) {
  return {(row % 8) * 4 + col % 4, (row / 8) * 4 + col / 4};
}

enum class FragmentMap { Operand, Result };

template <FragmentMap From, FragmentMap To, bool Transpose>
CUTE_HOST_DEVICE constexpr Owner remap_owner(int lane, int slot) {
  auto target = To == FragmentMap::Operand
      ? operand_coord(lane, slot) : result_coord(lane, slot);
  if constexpr (Transpose) {
    int const tmp = target.row; target.row = target.col; target.col = tmp;
  }
  return From == FragmentMap::Operand
      ? operand_owner(target.row, target.col)
      : result_owner(target.row, target.col);
}

template <FragmentMap From, FragmentMap To, bool Transpose, int Slot, int Source>
CUTE_HOST_DEVICE constexpr bool source_is_used() {
  for (int lane = 0; lane < 32; ++lane)
    if (remap_owner<From, To, Transpose>(lane, Slot).slot == Source) return true;
  return false;
}

template <FragmentMap From, FragmentMap To, bool Transpose>
CUTE_DEVICE void remap(uint32_t const* source, uint32_t* destination) {
#if defined(__HGGC_ARCH__)
  int const lane = int(threadIdx.x) % 32;
  uint32_t result[4];
  cute::for_each(cute::make_seq<4>{}, [&](auto word_index) {
    constexpr int word = decltype(word_index)::value;
    uint32_t packed = 0;
    cute::for_each(cute::make_seq<2>{}, [&](auto half_index) {
      constexpr int half = decltype(half_index)::value;
      auto const owner = remap_owner<From, To, Transpose>(lane, word * 2 + half);
      // The source slot varies with the DESTINATION lane, so every lane
      // exchanges each candidate before selecting. Selecting before shuffle
      // would use the source lane's slot and silently permute the matrix.
      uint32_t value = 0;
      cute::for_each(cute::make_seq<8>{}, [&](auto source_index) {
        constexpr int slot = decltype(source_index)::value;
        // These mappings use exactly two source slots per destination half;
        // eliminate the other six before code generation.
        if constexpr (source_is_used<From, To, Transpose, word * 2 + half, slot>()) {
          uint32_t const bits =
              (source[slot / 2] >> (16 * (slot % 2))) & 0xffffu;
          uint32_t const exchanged = __shfl_sync(0xffffffffu, bits, owner.lane);
          if (slot == owner.slot) value = exchanged;
        }
      });
      packed |= value << (half * 16);
    });
    result[word] = packed;
  });
  CUTE_UNROLL
  for (int word = 0; word < 4; ++word) destination[word] = result[word];
#else
  CUTE_INVALID_CONTROL_PATH("PPU fragment remap requires device execution");
#endif
}

template <class C, class B>
CUTE_DEVICE void result_to_b(C const& c, B& b) {
  static_assert(cute::size(C{}) == 8 && cute::size(B{}) == 8);
  remap<FragmentMap::Result, FragmentMap::Operand, true>(
      reinterpret_cast<uint32_t const*>(&c(0)),
      reinterpret_cast<uint32_t*>(&b(0)));
}

}  // namespace gdn_qsa::ppu
