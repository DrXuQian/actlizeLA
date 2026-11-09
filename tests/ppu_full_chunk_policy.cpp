#include <climits>
#include <iostream>

#include "gdn_qsa/ppu/wy_full_chunk.hpp"

using gdn_qsa::wy::gate_cache::full_chunks;
static_assert(!full_chunks(0) && !full_chunks(-64));
static_assert(!full_chunks(63) && full_chunks(64) && !full_chunks(65));
static_assert(full_chunks(128) && full_chunks(2048) && !full_chunks(2049));

int main() {
  using namespace gdn_qsa::wy;
  static_assert(measured_core_route(1, 2048, 16, 32) == CoreRoute::Solve8);
  static_assert(measured_core_route(1, 2048, 16, 64) == CoreRoute::Solve8);
  static_assert(measured_core_route(2, 8192, 16, 48) == CoreRoute::Solve8Value64);
  int route_checked = 0;
  for (int b : {0, 1, 2, 4}) for (int t : {0, 64, 512, 2048, 2049, 8192, 8193, 16384})
    for (int hk : {0, 8, 16, 32}) for (int hv : {0, 32, 48, 64}) {
      int const expected = hk == 16 && b == 1 && t == 2048 && (hv == 32 || hv == 64) ? 1 :
                           hk == 16 && b == 2 && t == 8192 && hv == 48 ? 2 : 0;
      if (int(measured_core_route(b, t, hk, hv)) != expected ||
          measured_core_route(b, t, hk, hv, false) != CoreRoute::Retained) return 1;
#if ALA_PPU10_BEST_CORE
      if (int(default_core_route({b, t, hk, hv})) != expected) return 1;
#else
      if (default_core_route({b, t, hk, hv}) != CoreRoute::Retained) return 1;
#endif
      ++route_checked;
    }
  std::cout << "固定core shape选路/显式OFF回退 PASS cases=" << route_checked << '\n';
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
