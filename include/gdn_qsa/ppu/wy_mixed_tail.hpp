#pragma once
#include "gdn_qsa/wy_contract.hpp"

namespace gdn_qsa::wy::mixed_tail {

struct Partition {
  int full_chunks;
  int tail_rows;
  constexpr explicit Partition(int sequence)
      : full_chunks(sequence > 0 ? sequence / Chunk : 0),
        tail_rows(sequence > 0 ? sequence % Chunk : 0) {}
  constexpr bool eligible() const { return full_chunks > 0 && tail_rows > 0; }
};

}  // namespace gdn_qsa::wy::mixed_tail
