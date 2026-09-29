#include <climits>
#include <iostream>

#include "gdn_qsa/ppu/wy_full_chunk.hpp"

using gdn_qsa::wy::gate_cache::full_chunks;
static_assert(!full_chunks(0) && !full_chunks(-64));
static_assert(!full_chunks(63) && full_chunks(64) && !full_chunks(65));
static_assert(full_chunks(128) && full_chunks(2048) && !full_chunks(2049));

int main() {
  int checked = 0;
  // Independent interval enumeration: every positive multiple of C64 is
  // eligible; every other member of each interval must use the tail path.
  for (int chunks = 0; chunks <= 512; ++chunks) {
    for (int remainder = 0; remainder < 64; ++remainder) {
      int const sequence = chunks * 64 + remainder;
      bool const expected = chunks > 0 && remainder == 0;
      if (full_chunks(sequence) != expected || full_chunks(-sequence)) {
        std::cerr << "full/tail eligibility mismatch: " << sequence << '\n';
        return 1;
      }
      checked += 2;
    }
  }
  if (full_chunks(INT_MIN) || full_chunks(INT_MAX) || !full_chunks(INT_MAX - 63))
    return 1;
  std::cout << "full/tail policy PASS cases=" << checked + 3 << '\n';
}
