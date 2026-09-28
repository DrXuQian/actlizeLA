#include <array>
#include <climits>
#include <cstdio>
#include <stdexcept>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_mixed_tail.hpp"
#include "gdn_qsa/ppu/wy_gate_coefficients.cuh"

using namespace gdn_qsa::wy;
using Tile = residual_warps8_hvlayout::StateTile;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
enum class Plant { None, SkipPrefix, DuplicateTail, WrongTail, MissingOwner, ShortDenominator };

uint64_t verify(Plant plant) {
  uint64_t bad = 0, lengths = 0, mixed = 0, full_steps = 0, tail_steps = 0;
  for (int length = 0; length <= 65536; ++length) {
    ++lengths;
    mixed_tail::Partition const part(length);
    bool const expected = length >= 65 && int64_t(length / 64) * 64 != length;
    bad += part.eligible() != expected;
    if (!part.eligible()) continue;
    ++mixed;
    int tokens = 0, count = 0;
    for (int ct = 0; ct < part.full_chunks - int(plant == Plant::SkipPrefix); ++ct) {
      bad += tokens != ct * 64 || residual::Plan::valid(ct, length) != 64;
      tokens += 64;
      ++count; ++full_steps;
    }
    int const tail_count = 1 + int(plant == Plant::DuplicateTail);
    for (int i = 0; i < tail_count; ++i) {
      bad += tokens != part.full_chunks * 64;
      int const valid = residual::Plan::valid(part.full_chunks, length) + int(plant == Plant::WrongTail);
      bad += valid != part.tail_rows || valid < 1 || valid >= 64;
      tokens += valid;
      ++count; ++tail_steps;
    }
    bad += tokens != length || count != (length + 63) / 64;
  }
  for (int length : {INT_MIN, -1, INT_MAX - 63, INT_MAX}) {
    mixed_tail::Partition const p(length);
    bad += p.eligible() != (length > 64 && int64_t(length / 64) * 64 != length);
    if (length > 0) bad += int64_t(p.full_chunks) * 64 + p.tail_rows != length;
  }
  // Actual native C ownership, all63 tail predicates. Invalid lanes still
  // publish defined zero padding; valid lanes must cover each output once.
  for (unsigned valid = 1; valid < 64; ++valid) {
    std::array<unsigned, 64*32> value{};
    std::array<unsigned, 128*32> state{};
    for (unsigned tid = 0; tid < Tile::Threads - unsigned(plant == Plant::MissingOwner); ++tid) {
      unsigned const warp = tid / 32, lane = tid % 32;
      for (int s = 0; s < 8; ++s) {
        unsigned const rc = Native::CLayout{}(lane, s);
        unsigned const row = Tile::value_row(warp, 0) + rc % 16;
        unsigned const col = Tile::column(warp) + rc / 16;
        if (row >= 64 || col >= 32) { ++bad; continue; }
        if (row < valid) ++value[row*32+col];
        for (int k = 0; k < Tile::KFragments; ++k) {
          unsigned const kr = Tile::k_row(warp, k) + rc % 16;
          if (kr >= 128) { ++bad; continue; }
          ++state[kr*32+col];
        }
      }
    }
    for (unsigned r = 0; r < 64; ++r)
      for (unsigned c = 0; c < 32; ++c) bad += value[r*32+c] != unsigned(r < valid);
    for (auto n : state) bad += n != 1;
  }
  if (plant == Plant::ShortDenominator) --lengths;
  constexpr uint64_t expected_mixed = 1023ull * 63;
  constexpr uint64_t expected_full_steps = 63ull * 1023 * 1024 / 2;
  bad += lengths != 65537 || mixed != expected_mixed ||
         full_steps != expected_full_steps || tail_steps != expected_mixed;
  if (plant == Plant::None)
    std::printf("[mixed tail host] lengths=%llu mixed=%llu full_steps=%llu tail_steps=%llu "
                "tail_extents=63 exact-once bad=%llu\n", (unsigned long long)lengths,
                (unsigned long long)mixed, (unsigned long long)full_steps,
                (unsigned long long)tail_steps, (unsigned long long)bad);
  return bad;
}

int main() {
  if (verify(Plant::None)) throw std::runtime_error("mixed-tail partition/owner mismatch");
  for (auto plant : {Plant::SkipPrefix, Plant::DuplicateTail, Plant::WrongTail, Plant::MissingOwner, Plant::ShortDenominator})
    if (!verify(plant)) throw std::runtime_error("mixed-tail negative escaped");
  std::puts("[mixed tail host] PASS negatives=5 device=NOT_RUN");
}
