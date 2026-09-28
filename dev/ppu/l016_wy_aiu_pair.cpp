#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_aiu.cuh"
#include "gdn_qsa/wy_contract.hpp"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
struct Counts { uint64_t cells = 0, values = 0, descriptors = 0; };

template <unsigned R, unsigned C>
uint64_t check_tile(Counts& counts, unsigned plant = 0) {
  using T = aiu::Tile<R, C>;
  std::vector<uint32_t> memory(R * C / 2);
  std::vector<unsigned> owners(R * C);
  uint64_t bad = 0;
  for (unsigned r = 0; r < R; ++r) for (unsigned c = 0; c < C; ++c) {
    unsigned at = T::offset(r, c);
    if (plant == 1) at ^= 8u;  // same extents, wrong swizzle
    if (plant == 2) at = unsigned(swizzle<R, C>(r, c));  // old16x16 placement
    ++owners.at(at);
    memory.at(at / 2) |= (r * C + c + 1) << (16 * (at % 2));
  }
  for (auto n : owners) bad += n != 1;
  for (unsigned cube = 0; cube < T::Cubes - unsigned(plant == 3); ++cube) {
    for (unsigned row = 0; row < R; row += 16) {
      for (unsigned col = 0; col < T::CubeWidth; col += 16) {
        for (unsigned lane = 0; lane < 32; ++lane) {
          threadIdx.x = lane;
          uint32_t words[4];
          // Execute the REAL actlize load simulator. It is independent of the
          // candidate's placement helper and includes the cross-slice rotation.
          cute::ppu_tsm_ld_swzl_sim<BF16, R, T::CubeWidth, true>(words,
              memory.data() + cube * R * T::CubeWidth / 2, col, row, 0);
          for (unsigned slot = 0; slot < 8; ++slot) {
            unsigned const logical = Native::ALayout{}(lane, slot);
            unsigned const r = row + logical % 16;
            unsigned const c = cube * T::CubeWidth + col + logical / 16;
            unsigned const actual = (words[slot / 2] >> (16 * (slot % 2))) & 0xffffu;
            bad += actual != r * C + c + 1;
            ++counts.values;
          }
        }
      }
    }
  }
  for (unsigned r = 0; r < R; ++r) for (unsigned c = 0; c < C; c += 8) {
    unsigned const base = T::offset(r, c);
    bad += base % 8 != 0;
    for (unsigned e = 0; e < 8; ++e) bad += T::offset(r, c + e) != base + e;
  }
  // The same descriptors consumed by the real AIU writer: every legal tail,
  // contiguous and head-strided input, all logical cells including padz.
  for (int stride : {int(C), 128, 256, 512, 2048, 4096, 8192}) for (unsigned valid = 1; valid <= R; ++valid) {
    auto desc = T::descriptor(stride, valid);
    if (plant == 4) desc.dim_w *= 2;  // bytes mistaken for BF16 elements
    if (plant == 5) ++desc.dim_h;     // cross-chunk tail read
    bad += desc.cube_h != R || desc.cube_w != T::CubeWidth || desc.offset_w != 0;
    for (unsigned r = 0; r < R; ++r) for (unsigned c = 0; c < C; ++c) {
      bool const present = r < unsigned(desc.dim_h);
      bad += present != (r < valid);
      if (present) bad += int64_t(r) * desc.dim_w + c != int64_t(r) * stride + c;
      ++counts.descriptors;
    }
  }
  bad += !T::admitted_stride(C) || T::admitted_stride(C - 1) ||
         T::admitted_stride(int64_t(INT_MAX) + 1) || T::admitted_stride(int64_t(1) << 33);
  ++counts.cells;
  return bad;
}

uint64_t suite(Counts& counts, unsigned plant = 0) {
  uint64_t bad = check_tile<16, 16>(counts, plant) + check_tile<16, 32>(counts, plant) +
      check_tile<16, 64>(counts, plant) + check_tile<64, 128>(counts, plant) +
      check_tile<64, 32>(counts, plant) + check_tile<128, 32>(counts, plant) +
      check_tile<64, 64>(counts, plant) + check_tile<128, 64>(counts, plant) +
      check_tile<64, 16>(counts, plant) + check_tile<128, 16>(counts, plant);
  // Pin the full inventory independently; omitting a cube cannot lower the denominator.
  bad += counts.cells != 10 || counts.values != 31488;
  bad += counts.descriptors != 7ull * (16 * (256 + 512 + 1024) +
      64 * (8192 + 2048 + 4096 + 1024) + 128 * (4096 + 8192 + 2048));
  return bad;
}

unsigned selection(bool plant = false) {
  unsigned bad = 0, admitted = 0, calls = 0;
  for (unsigned mask = 4096; mask < 16384; ++mask) {
    bool const expected = mask == 5616 || mask == 9712 || mask == 13808;
    bool const actual = valid_delivery(mask);
    admitted += actual;
    bad += actual != expected;
    if (actual) {
      unsigned const options = plant ? AiuState : mask & AiuOptions;
      int const got = visit_aiu_options(options, [&](auto state, auto output) {
        ++calls;
        return (decltype(state)::value ? 4096 : 0) | (decltype(output)::value ? 8192 : 0);
      }, -1);
      bad += got != int(mask & AiuOptions);
    }
  }
  bad += admitted != 3 || calls != 3;
  bad += visit_aiu_options(0, [](auto, auto) { return 1; }, -1) != -1;
  return bad;
}

int main() {
  Counts counts;
  if (suite(counts) || selection()) throw std::runtime_error("AIU paired layout/descriptor/selector mismatch");
  for (unsigned plant = 1; plant <= 6; ++plant) {
    Counts scratch;
    auto const bad = plant == 6 ? selection(true) : suite(scratch, plant);
    if (!bad) throw std::runtime_error("AIU negative escaped");
    std::printf("[WY AIU negative] plant=%u bad=%llu EXPECTED-RED/PASS\n", plant, (unsigned long long)bad);
  }
  std::printf("[WY AIU pair] shapes=%llu values=%llu descriptor_elements=%llu selectors=3 "
              "authority=actlize-load-simulator+native-MMA-traits PASS device_execution=NOT_RUN\n",
              (unsigned long long)counts.cells, (unsigned long long)counts.values,
              (unsigned long long)counts.descriptors);
}
