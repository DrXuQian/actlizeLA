#pragma once

#include <cute/config.hpp>

namespace gdn_qsa::wy {

// Nonnegative logical coordinates only. This is the existing RowLayout,
// expressed without signed division/remainder; l010 checks every element
// against both that layout and an independent hardware-cube formula.
template <unsigned Rows, unsigned Cols>
CUTE_HOST_DEVICE constexpr unsigned state_shared_offset(unsigned row, unsigned col) {
  static_assert(Rows % 16 == 0 && Cols % 16 == 0);
  unsigned const linear = (row & 15u) * 16u + (row >> 4) * 256u +
                          (col & 15u) + (col >> 4) * (Rows * 16u);
  return linear ^ ((linear >> 3) & 8u);
}

template <unsigned Rows, unsigned Cols, unsigned ElementBytes, unsigned Threads>
struct StateVectorPlan {
  static constexpr unsigned Width = 16 / ElementBytes;
  static constexpr unsigned Count = Rows * Cols / Width;
  static constexpr unsigned Iterations = Count / Threads;
  static_assert(ElementBytes == 2 || ElementBytes == 4);
  static_assert(Cols % Width == 0 && Count % Threads == 0);
  CUTE_HOST_DEVICE static constexpr unsigned vector(unsigned tid, unsigned iteration) {
    return tid + iteration * Threads;
  }
  CUTE_HOST_DEVICE static constexpr unsigned row(unsigned vector) { return vector / (Cols / Width); }
  CUTE_HOST_DEVICE static constexpr unsigned col(unsigned vector) { return (vector % (Cols / Width)) * Width; }
};

struct StateGateRows {
  static constexpr int Count = 2;
  CUTE_HOST_DEVICE static constexpr int half(int slot) { return slot / 4; }
  CUTE_HOST_DEVICE static constexpr int row(int lane, int half) { return lane / 4 + 8 * half; }
};

}  // namespace gdn_qsa::wy
