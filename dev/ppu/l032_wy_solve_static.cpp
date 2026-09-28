// Execute the actual template helper, with an independent rolled oracle.
#include "gdn_qsa/ppu/wy_solve_static.cuh"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using Matrix = std::array<float, 64 * 64>;
bool same(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }
void require(bool ok, char const* why) { if (!ok) throw std::runtime_error(why); }

// No shared indexing/helper calls. The negative variants are intentionally
// wrong versions of this independent oracle, never device candidates.
void reference(Matrix const& lower, unsigned warp, unsigned lane, Matrix& inverse,
               float (&column)[16], int plant = 0) {
  for (int r = 0; r < 16; ++r) {
    float x = r == int(lane % 16) ? 1.0f : 0.0f;
    for (int j = 0; j < r; ++j) {
      int const k = plant == 2 ? r - 1 - j : j;
      unsigned const row = warp * 16 + r, col = warp * 16 + k;
      int const pick = plant == 1 && k > 0 ? k - 1 : k;
      if (!(plant == 3 && j == r - 1)) x -= lower[row * 64 + col] * column[pick];
    }
    column[r] = x;
    if (lane < 16 && !(plant == 4 && r == 15))
      inverse[(warp * 16 + r) * 64 + warp * 16 + lane] = x;
  }
}
}

int main(int argc, char** argv) {
  try {
    bool const omit_last = argc == 2 && std::strcmp(argv[1], "--omit-last-context") == 0;
    require(argc == 1 || omit_last, "unexpected host-gate option");
    uint32_t rng = 0x9e3779b9u;
    uint64_t contexts = 0, values = 0, products = 0, published = 0;
    std::array<uint64_t, 4> detected{};
    for (unsigned tail = 1; tail <= 64; ++tail) {
      for (unsigned pattern = 0; pattern < 8; ++pattern) {
        Matrix lower{};
        for (unsigned r = 0; r < 64; ++r) for (unsigned k = 0; k < 64; ++k) {
          rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
          // Dense, signed, non-power-of-two coefficients plus tail identities.
          lower[r * 64 + k] = r < tail && k < r
              ? (int(rng % 2001) - 1000) / (10003.0f + float(pattern * 19)) : 0.0f;
        }
        for (unsigned warp = 0; warp < 4; ++warp) for (unsigned lane = 0; lane < 32; ++lane) {
          if (omit_last && tail == 64 && pattern == 7 && warp == 3 && lane == 31) continue;
          Matrix got, want; got.fill(-12345.25f); want = got;
          float actual[16] = {}, expected[16] = {};
          unsigned const base = warp * 16 * 65;
          gdn_qsa::wy::solve_static::diagonal(lower.data() + base, got.data() + base, lane, actual);
          reference(lower, warp, lane, want, expected);
          for (unsigned r = 0; r < 16; ++r) {
            require(same(actual[r], expected[r]), "diagonal value/order differs");
            ++values; products += r;
          }
          require(std::memcmp(got.data(), want.data(), sizeof(Matrix)) == 0,
                  "publisher touched wrong row/lane/warp or guard");
          published += lane < 16 ? 16 : 0;
          ++contexts;
          for (int plant = 1; plant <= 4; ++plant) {
            Matrix wrong; wrong.fill(-12345.25f); float bad[16] = {};
            reference(lower, warp, lane, wrong, bad, plant);
            detected[plant - 1] += std::memcmp(bad, actual, sizeof bad) != 0 ||
                                  std::memcmp(wrong.data(), got.data(), sizeof(Matrix)) != 0;
          }
        }
      }
    }
    require(contexts == 65536 && values == 1048576 && products == 7864320 && published == 524288,
            "coverage denominator incomplete");
    for (unsigned i = 0; i < detected.size(); ++i) {
      require(detected[i] != 0, "wrong index/order/term/publisher negative escaped");
      std::cout << "[solve static host negative] plant=" << i+1 << " mismatches=" << detected[i]
                << " EXPECTED-RED/PASS\n";
    }
    std::cout << "[solve static host] contexts=" << contexts << " values=" << values
              << " ordered_products=" << products << " stores=" << published
              << " RAW-BIT/PASS native-contraction=SEPARATE-GATE device=NOT_RUN\n";
  } catch (std::exception const& e) { std::cerr << e.what() << '\n'; return 1; }
}
