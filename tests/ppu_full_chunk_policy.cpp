#include <climits>
#include <iostream>

#include "gdn_qsa/ppu/wy_full_chunk.hpp"
#include "gdn_qsa/ppu/wy_dispatch.hpp"

using gdn_qsa::wy::gate_cache::full_chunks;
static_assert(!full_chunks(0) && !full_chunks(-64));
static_assert(!full_chunks(63) && full_chunks(64) && !full_chunks(65));
static_assert(full_chunks(128) && full_chunks(2048) && !full_chunks(2049));

int main() {
  using gdn_qsa::wy::CoreRoute;
  using gdn_qsa::wy::core_route;
  using gdn_qsa::wy::native_bf16_solve_threads;
  int native_cases = 0;
  for (int b : {0, 1, 2, 4})
    for (int t : {1, 63, 64, 65, 127, 128, 129, 2048, 2049, 8192, 8193, 16384})
      for (int hk : {1, 2, 16, 32})
        for (int hv : {2, 16, 32, 48, 64})
          for (bool fp32 : {false, true}) {
            bool const expected = fp32 && b == 1 && hk == 16 &&
                (t == 2048 || t == 8192) && (hv == 32 || hv == 48);
            if ((core_route({b, t, hk, hv}, fp32) == CoreRoute::NativeBf16) != expected)
              return 1;
            if (expected) {
              ++native_cases;
              unsigned const threads = t == 2048 && hv == 32 ? 256 : 128;
              if (native_bf16_solve_threads({b, t, hk, hv}) != threads) return 1;
            }
          }
  if (native_cases != 4) return 1;  // Gate values do not change this denominator.
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
  std::cout << "full/tail policy PASS cases=" << checked + 3
            << " native-bf16-shapes=" << native_cases << '\n';
}
