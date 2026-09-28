#include <array>
#include <cstdio>
#include <stdexcept>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_full_chunk.hpp"
#include "gdn_qsa/ppu/wy_tiles.cuh"
#include "gdn_qsa/ppu/wy_split_prepare.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
enum class Plant { None, MissingLane, LostCausalDiagonal, ShortDenominator };

uint64_t verify(Plant plant) {
  uint64_t bad = 0, chunks = 0, lengths = 0;
  for (int length = 0; length <= 65536; ++length) {
    ++lengths;
    bool const oracle = length > 0 && int64_t(length / 64) * 64 == length;
    bool const admitted = gate_cache::full_chunks(length);
    bad += admitted != oracle;
    if (!admitted) continue;
    Shape shape{1, length, 1, 2};
    for (int ct = 0; ct < shape.chunks(); ++ct) {
      ++chunks;
      for (int row = 0; row < 64; ++row) {
        bad += int64_t(ct) * 64 + row >= length;
        // Old beta clamp/predicate is an identity at the admitted extent.
        int const old_row = row < 64 ? row : 63;
        bad += old_row != row;
      }
    }
  }
  std::array<unsigned, 4096> lower{}, attention{}, inverse{};
  std::array<unsigned, 8192> output{};
  uint64_t lower_visits = 0, output_visits = 0, causal = 0, strict = 0;
  for (unsigned tid = 0; tid < split_prepare::Plan::SolveThreads; ++tid) {
    unsigned const warp = tid / 32, lane = tid % 32;
    for (unsigned tile = warp; tile < 16; tile += 4) {
      unsigned const br = tile / 4, bc = tile % 4;
      if (bc > br) continue;
      for (unsigned slot = 0; slot < 8; ++slot) {
        unsigned const offset = Native::CLayout{}(lane, slot);
        unsigned const r = br * 16 + offset % 16, c = bc * 16 + offset / 16;
        if (r >= 64 || c >= 64) { ++bad; continue; }
        ++lower[r * 64 + c];
        ++lower_visits;
        strict += r > c;
      }
    }
    for (unsigned it = 0; it < split_prepare::Plan::InverseIterations; ++it)
      for (unsigned word = 0; word < split_prepare::Plan::VectorElements; ++word) {
        unsigned const i = (tid + it * split_prepare::Plan::SolveThreads) * split_prepare::Plan::VectorElements + word;
        if (i >= inverse.size()) { ++bad; continue; }
        ++inverse[i];
      }
  }
  for (unsigned tid = 0; tid < OutputTile::Threads - unsigned(plant == Plant::MissingLane); ++tid) {
    unsigned const warp = tid / 32, lane = tid % 32;
    for (int fragment = 0; fragment < OutputTile::Fragments; ++fragment) {
      for (unsigned slot = 0; slot < 8; ++slot) {
        unsigned const offset = Native::CLayout{}(lane, slot);
        unsigned const row = OutputTile::row(warp) + offset % 16;
        unsigned const col = OutputTile::column(warp, fragment) + offset / 16;
        if (row >= 64 || col >= 64) { ++bad; continue; }
        ++attention[row * 64 + col];
        bool const keep = plant == Plant::LostCausalDiagonal ? row > col : row >= col;
        bad += keep != (int(row) >= int(col) && row < 64);
        causal += keep;
        for (int panel = 0; panel < Dim; panel += OutputTile::Panel) {
          ++output[row * 128 + panel + col];
          ++output_visits;
        }
      }
    }
  }
  for (unsigned i = 0; i < 4096; ++i) {
    bad += lower[i] != unsigned((i / 64) / 16 >= (i % 64) / 16);
    bad += attention[i] != 1 || inverse[i] != 1;
  }
  for (unsigned count : output) bad += count != 1;
  if (plant == Plant::ShortDenominator) --lengths;
  bad += lengths != 65537 || chunks != 524800 || lower_visits != 2560 || strict != 2016 ||
         causal != 2080 || output_visits != 8192;
  if (plant == Plant::None)
    std::printf("[full stages host] lengths=65537 chunks=524800 lower=2560 strict=2016 "
                "inverse=4096 attention=4096 causal=2080 output=8192 bad=%llu\n", (unsigned long long)bad);
  return bad;
}

int main() {
  if (verify(Plant::None)) throw std::runtime_error("full stage extent/ownership/mask mismatch");
  for (auto plant : {Plant::MissingLane, Plant::LostCausalDiagonal, Plant::ShortDenominator})
    if (!verify(plant)) throw std::runtime_error("full stage negative escaped");
  std::puts("[full stages host] PASS negatives=3 device=NOT_RUN");
}
