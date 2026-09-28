#include <array>
#include <climits>
#include <cstdio>
#include <stdexcept>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_full_chunk.hpp"
#include "gdn_qsa/ppu/wy_gate_coefficients.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
using Tile = residual_warps8_hvlayout::StateTile;
enum class Plant { None, TailEligible, MissingLane, ShortDenominator };

uint64_t suite(Plant plant) {
  uint64_t bad = 0, lengths = 0, full = 0, cells = 0;
  // Exhaust this declared range, plus both signed extent boundaries below.
  // Positive length=64*q+r is full iff r=0; no multiply/add in admission.
  for (int length = 0; length <= 65536; ++length) {
    bool const selected = gate_cache::full_chunks(length) ||
        (plant == Plant::TailEligible && length == 65);
    bool const oracle = length > 0 && (length / 64) * 64 == length;
    bad += selected != oracle;
    ++lengths;
    if (!selected) continue;
    ++full;
    Shape shape{1, length, 1, 2};
    for (int ct = 0; ct < shape.chunks(); ++ct) {
      bad += residual::Plan::valid(ct, length) != 64;
      bad += int64_t(ct) * 64 + 63 >= length;
      ++cells;
    }
  }
  for (int length : {INT_MIN, -64, -1, INT_MAX - 63, INT_MAX})
    bad += gate_cache::full_chunks(length) !=
        (length > 0 && int64_t(length / 64) * 64 == length);
  std::array<unsigned, 64 * 32> values{};
  std::array<unsigned, 128 * 32> state{};
  std::array<unsigned, 64> beta{};
  uint64_t threads = 0;
  for (unsigned tid = 0; tid < Tile::Threads - unsigned(plant == Plant::MissingLane); ++tid) {
    ++threads;
    if (tid < 64) ++beta[tid];
    unsigned const warp = tid / 32, lane = tid % 32;
    for (int s = 0; s < 8; ++s) {
      unsigned const native = Native::CLayout{}(lane, s);
      unsigned const col = Tile::column(warp) + native / 16;
      unsigned const row = Tile::value_row(warp, 0) + native % 16;
      if (row >= 64 || col >= 32) { ++bad; continue; }
      ++values[row * 32 + col];
      for (int k = 0; k < Tile::KFragments; ++k) {
        unsigned const kr = Tile::k_row(warp, k) + native % 16;
        if (kr >= 128) { ++bad; continue; }
        ++state[kr * 32 + col];
      }
    }
  }
  for (auto n : values) bad += n != 1;
  for (auto n : state) bad += n != 1;
  for (auto n : beta) bad += n != 1;
  if (plant == Plant::ShortDenominator) --lengths;
  bad += lengths != 65537 || full != 1024 || cells != 524800 || threads != 256;
  if (plant == Plant::None)
    std::printf("[full chunk host] lengths=%llu full=%llu chunks=%llu values=2048 state=4096 beta=64 bad=%llu\n",
        (unsigned long long)lengths, (unsigned long long)full,
        (unsigned long long)cells, (unsigned long long)bad);
  return bad;
}

int main() {
  if (suite(Plant::None)) throw std::runtime_error("full-chunk admission/ownership failed");
  for (auto p : {Plant::TailEligible, Plant::MissingLane, Plant::ShortDenominator})
    if (!suite(p)) throw std::runtime_error("full-chunk negative escaped");
  std::puts("[full chunk host] PASS negatives=3 device=NOT_RUN");
}
