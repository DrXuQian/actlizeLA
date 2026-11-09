#include "gdn_qsa/ppu/wy_parallel_wu.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

using Plan = gdn_qsa::wy::parallel_wu::WorkspacePlan;

bool check(bool wrong_pitch, bool missing_group) {
  constexpr unsigned groups = 4;
  std::vector<unsigned> memory(groups * 16384, 0), owners(memory.size(), 0);
  for (unsigned g = 0; g < groups; ++g)
    for (unsigned i = 0; i < 4096; ++i) memory[g * 16384 + i] = 10000 + g * 4096 + i;
  // Deliberately nonsequential CTA completion. Each P must still be intact
  // when its owner starts, regardless of other groups already publishing W/U.
  std::array<unsigned, 4> order{1, 3, 0, 2};
  unsigned reads = 0, writes = 0;
  for (unsigned g : order) {
    if (missing_group && g == 2) continue;
    for (unsigned i = 0; i < 4096; ++i) {
      if (memory[g * 16384 + i] != 10000 + g * 4096 + i) return false;
      ++reads;
    }
    auto const w = wrong_pitch ? g * 8192 : Plan::w(g);
    auto const u = wrong_pitch ? w + 8192 : Plan::u(g);
    for (unsigned i = 0; i < 8192; ++i) {
      memory.at(w + i) = 100000 + g * 16384 + i;
      memory.at(u + i) = 100000 + g * 16384 + 8192 + i;
      ++owners.at(w + i); ++owners.at(u + i); writes += 2;
    }
  }
  if (reads != 16384 || writes != 65536) return false;
  for (unsigned g = 0; g < groups; ++g)
    for (unsigned i = 0; i < 16384; ++i)
      if (memory[g * 16384 + i] != 100000 + g * 16384 + i || owners[g * 16384 + i] != 1) return false;
  return true;
}

int main() {
  assert(check(false, false));
  assert(!check(true, false));
  assert(!check(false, true));
  std::puts("[parallel WU workspace] per-group P->W/U ownership PASS; packed-pitch and missing-group EXPECTED_RED");
}
