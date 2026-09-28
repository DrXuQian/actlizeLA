#include <array>
#include <cstdio>
#include <stdexcept>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_first_chunk.hpp"
#include "gdn_qsa/ppu/wy_residual_warps8_hvlayout.cuh"

using namespace gdn_qsa::wy;
namespace Layout = residual_warps8_hvlayout;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
enum class Plant { None, SuppliedState, SecondChunk, MissingOwner, ShortDenominator };

uint64_t verify(Plant plant) {
  uint64_t bad = 0, extent_cases = 0, chunks = 0, skipped = 0;
  for (int length = 0; length <= 65536; ++length) {
    for (bool initial : {false, true}) {
      ++extent_cases;
      bool admitted = first_chunk::eligible(length, initial);
      if (plant == Plant::SuppliedState) admitted = gate_cache::full_chunks(length);
      bool const expected = !initial && length > 0 && (length / 64) * 64 == length;
      bad += admitted != expected;
      if (!admitted) continue;
      for (int ct = 0; ct < length / 64; ++ct) {
        ++chunks;
        bool const history = plant == Plant::SecondChunk ? ct > 1 : first_chunk::has_history(ct);
        bad += history != (int64_t(ct) * 64 != 0);
        skipped += !history;
      }
    }
  }
  // The first H snapshot is still initialized/published. Count ownership
  // with distinct tickets, not zero data that could conceal duplicate writes.
  std::array<unsigned, 16384> snapshot{};
  std::array<unsigned, 8192> output{};
  for (unsigned slice = 0; slice < 4; ++slice)
    for (unsigned tid = 0; tid < Layout::Plan::Threads; ++tid) {
      auto const warp = tid / 32, lane = tid % 32;
      if (plant == Plant::MissingOwner && tid == 255) continue;
      for (unsigned k = 0; k < Layout::StateTile::KFragments; ++k)
        for (unsigned s = 0; s < 8; ++s) {
          unsigned const rc = Native::CLayout{}(lane, s);
          unsigned const row = Layout::StateTile::k_row(warp, k) + rc % 16;
          unsigned const col = slice * 32 + Layout::StateTile::column(warp) + rc / 16;
          if (row >= 128 || col >= 128) { ++bad; continue; }
          ++snapshot[row * 128 + col];
        }
    }
  for (unsigned tid = 0; tid < OutputTile::Threads; ++tid)
    for (int panel = 0; panel < 128; panel += OutputTile::Panel)
      for (unsigned c = 0; c < OutputTile::Fragments; ++c)
        for (unsigned s = 0; s < 8; ++s) {
          unsigned const rc = Native::CLayout{}(tid % 32, s);
          unsigned const row = OutputTile::row(tid / 32) + rc % 16;
          unsigned const col = panel + OutputTile::column(tid / 32, c) + rc / 16;
          if (row >= 64 || col >= 128) { ++bad; continue; }
          ++output[row * 128 + col];
        }
  for (auto count : snapshot) bad += count != 1;
  for (auto count : output) bad += count != 1;
  if (plant == Plant::ShortDenominator) --extent_cases;
  bad += extent_cases != 131074 || chunks != 524800 || skipped != 1024;
  // Work denominator from logical dimensions, independent of source loops.
  constexpr uint64_t heads = 32, c = 64, k = 128, v = 128, nt = 32, atom = 16*16*16;
  constexpr uint64_t kh = heads*c*k*v/atom, qh = kh;
  constexpr uint64_t state = heads*nt*(c*k*v + c*c*v + k*c*v)/atom;
  constexpr uint64_t out = heads*nt*(c*c*k + c*k*v + c*c*v)/atom;
  static_assert(kh == 8192 && qh == 8192 && state == 655360 && out == 524288);
  if (plant == Plant::None)
    std::printf("[first chunk host] extent_cases=%llu chunks=%llu skipped=%llu "
                "H_owners=16384 O_owners=8192 predicted_state_mma=647168 "
                "predicted_output_mma=516096 bad=%llu\n",
                (unsigned long long)extent_cases, (unsigned long long)chunks,
                (unsigned long long)skipped, (unsigned long long)bad);
  return bad;
}
int main() {
  if (verify(Plant::None)) throw std::runtime_error("first chunk coverage mismatch");
  for (auto plant : {Plant::SuppliedState, Plant::SecondChunk, Plant::MissingOwner, Plant::ShortDenominator})
    if (!verify(plant)) throw std::runtime_error("first chunk negative escaped");
  std::puts("[first chunk host] PASS negatives=4 device=NOT_RUN");
}
