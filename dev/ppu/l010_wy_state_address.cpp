#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_tiles.cuh"
#include "gdn_qsa/ppu/wy_state_address.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;

// Independent native-cube anchor. Do not repeat the candidate's XOR formula.
constexpr unsigned hardware(unsigned r, unsigned c, unsigned rows) {
  unsigned const cube = (c / 16) * (rows / 16) + r / 16;
  r %= 16; c %= 16;
  unsigned const line = r / 4, vec = ((r % 4) * 2 + c / 8) ^ (line % 2);
  return cube * 256 + (line * 32 + vec * 4 + (c % 8) / 2) * 2 + c % 2;
}

struct Counts { uint64_t coordinates = 0, vectors = 0, gate_values = 0; };

template <int Rows, int Cols, int ElementBytes = 2>
uint64_t addresses(Counts& count, int plant = 0) {
  using Plan = StateVectorPlan<Rows, Cols, ElementBytes, 128>;
  uint64_t bad = 0;
  for (unsigned r = 0; r < Rows; ++r) {
    for (unsigned c = 0; c < Cols; ++c) {
      if constexpr (ElementBytes == 2) {
        auto const actual = state_shared_offset<Rows, Cols>(r, c) ^ (plant == 1 ? 8u : 0u);
        bad += actual != unsigned(swizzle<Rows, Cols>(int(r), int(c)));
        bad += actual != hardware(r, c, Rows);
      }
      ++count.coordinates;
    }
  }
  for (int64_t stride : {int64_t(Cols), int64_t(Cols + 8), int64_t(2048), int64_t(4096), int64_t(1) << 33}) {
    for (unsigned valid = 0; valid <= Rows; ++valid) {
      std::vector<int> owners(Rows * Cols);
      for (unsigned tid = 0; tid < 128; ++tid) {
        for (unsigned it = 0; it < Plan::Iterations - (plant == 3 ? 1u : 0u); ++it) {
          unsigned const i = Plan::vector(tid, it), r = Plan::row(i), c = Plan::col(i);
          unsigned const flat = tid * Plan::Width + it * 128 * Plan::Width;
          bad += r != flat / Cols || c != flat % Cols;
          bool const active = plant == 4 ? r <= valid : r < valid;
          int64_t const source = active ? int64_t(r) * stride * (plant == 2 ? ElementBytes : 1) + c : 0;
          int64_t const expected = flat / Cols < valid ? int64_t(flat / Cols) * stride + flat % Cols : 0;
          bad += active != (flat / Cols < valid) || source != expected;
          unsigned const shared = ElementBytes == 2 ? state_shared_offset<Rows, Cols>(r, c) : r * Cols + c;
          bad += (shared * ElementBytes) % 16 != 0;
          for (unsigned x = 0; x < Plan::Width; ++x) {
            unsigned const physical = ElementBytes == 2 ? hardware(r, c + x, Rows) : r * Cols + c + x;
            bad += shared + x != physical;
            ++owners.at(r * Cols + c + x);
          }
          ++count.vectors;
        }
      }
      for (int n : owners) bad += n != 1;
    }
  }
  return bad;
}

uint64_t gates(Counts& count, bool plant = false) {
  uint64_t bad = 0;
  // Nonconstant gates distinguish all rows, including rows of a padded tail.
  for (int fixture = 0; fixture < 64; ++fixture) {
    float g[64];
    for (int r = 0; r < 64; ++r) g[r] = -float((r * 37 + fixture * 11) % 251) / 16.0f;
    float const last = g[fixture];
    for (int warp = 0; warp < 4; ++warp) for (int lane = 0; lane < 32; ++lane) {
      for (int frag = 0; frag < StateTile::ValueFragments; ++frag) {
        float cached[2];
        for (int half = 0; half < 2; ++half) {
          int const row = StateTile::value_row(warp, frag) + StateGateRows::row(lane, half);
          cached[half] = std::exp(last - g[row]);
        }
        for (int slot = 0; slot < 8; ++slot) {
          int const native_row = Native::CLayout{}(lane, slot) % 16;
          int const row = StateTile::value_row(warp, frag) + native_row;
          int const half = StateGateRows::half(slot) ^ int(plant);
          float const control = std::exp(last - g[row]);
          bad += std::memcmp(&control, &cached[half], sizeof(float)) != 0;
          // The same arguments, not an approximate exp2 replacement.
          bad += native_row != StateGateRows::row(lane, StateGateRows::half(slot));
          ++count.gate_values;
        }
      }
    }
  }
  return bad;
}

int dispatch(bool plant = false) {
  int bad = 0;
  for (unsigned options = 0; options < 512; ++options) {
    int calls = 0;
    int const got = visit_state_options(plant ? options & 64u : options, [&](auto address, auto gates) {
      ++calls;
      return (decltype(address)::value ? 64 : 0) | (decltype(gates)::value ? 128 : 0);
    }, -73);
    bool const valid = options == 64 || options == 128 || options == 192;
    bad += got != (valid ? int(options) : -73) || calls != int(valid);
  }
  return bad;
}

int main() {
  Counts count;
  if (addresses<64, 128>(count) || addresses<64, 32>(count) ||
      addresses<128, 32>(count) || addresses<128, 32, 4>(count) || gates(count) || dispatch()) return 1;
  constexpr uint64_t vectors = 5ull * (65 * 1024 + 65 * 256 + 129 * 512 + 129 * 1024);
  if (count.coordinates != 18432 || count.vectors != vectors || count.gate_values != 131072)
    throw std::runtime_error("state-copy/gates denominator differs");
  for (int plant = 1; plant <= 6; ++plant) {
    Counts scratch;
    uint64_t const bad = plant == 6 ? dispatch(true) : plant == 5 ? gates(scratch, true) : addresses<64, 128>(scratch, plant);
    if (!bad) throw std::runtime_error("state-address negative escaped");
    std::printf("[WY state-address negative] plant=%d bad=%llu EXPECTED-RED/PASS\n", plant,
                static_cast<unsigned long long>(bad));
  }
  std::printf("[WY state-address] coordinates=%llu vectors=%llu gate_values=%llu typed-dispatch=512/3 independent-native-anchor=PASS device_execution=NOT_RUN\n",
      static_cast<unsigned long long>(count.coordinates), static_cast<unsigned long long>(count.vectors),
      static_cast<unsigned long long>(count.gate_values));
}
