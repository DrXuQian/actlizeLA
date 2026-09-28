#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_warps8_blayout.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
using Eight = residual_warps8::StateTile;
using B = residual_warps8_blayout::BIntermediate;
static_assert(std::is_same_v<residual_warps8::Storage, residual_warps8_blayout::Storage>);
static_assert(std::is_same_v<Eight, residual_warps8_blayout::StateTile>);

struct Counts { uint64_t writes = 0, reads = 0, out_of_bounds = 0; };
enum Plant { None, StaleFourWarpBase, StaleValueLayout, WrongSwizzle,
             WrongCube, MissingLane, MissingWarp, MissingKTile, MissingPlane };

// uint16 tags are independent logical (plane,slice,row,col) identities.
unsigned tag(unsigned plane, unsigned slice, unsigned row, unsigned col, unsigned valid) {
  return row < valid ? 1 + plane * 8192 + slice * 2048 + row * ValueTile + col : 0;
}

unsigned check(Counts& counts, unsigned valid, unsigned slice, unsigned plane, Plant plant = None) {
  unsigned bad = 0;
  std::vector<uint32_t> memory(Chunk * ValueTile / 2);
  std::vector<unsigned> owners(Chunk * ValueTile);
  for (unsigned warp = 0; warp < Eight::Threads / 32 - unsigned(plant == MissingWarp); ++warp)
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned f = 0; f < Eight::ValueFragments; ++f)
        for (unsigned slot = 0; slot < 8; ++slot) {
          unsigned const native = Native::CLayout{}(lane, slot);
          unsigned const row = Eight::value_row(warp, f) + native % 16;
          unsigned const col = Eight::column(warp) + native / 16;
          unsigned base = B::producer_base(warp, lane);
          if (plant == StaleFourWarpBase) base = residual_blayout::BIntermediate::producer_base(warp, lane);
          unsigned at = B::producer_offset(base, f, slot);
          if (plant == StaleValueLayout) at = residual::Value::offset(row, col);
          if (plant == WrongSwizzle) at ^= 8;
          bad += at != B::offset(row, col);
          ++counts.writes;
          if (at >= owners.size()) { ++bad; ++counts.out_of_bounds; continue; }
          ++owners[at];
          unsigned const shift = 16 * (at % 2);
          memory[at / 2] = (memory[at / 2] & ~(65535u << shift)) |
                          (tag(plane, slice, row, col, valid) << shift);
        }
  for (auto count : owners) bad += count != 1;
  // Enumerate each actual consumer warp, every K tile and all B fragments.
  // Both P@R and Kt@scaledV use this map. The real actlize native-load
  // simulator reads the bytes; it does not invert our writer's offset helper.
  for (unsigned warp = 0; warp < Eight::Threads / 32; ++warp)
    for (unsigned row = 0; row < Chunk - 16 * unsigned(plant == MissingKTile); row += 16)
      for (unsigned lane = 0; lane < 32 - unsigned(plant == MissingLane); ++lane) {
        threadIdx.x = lane;
        unsigned cube = B::cube(row, Eight::column(warp));
        if (plant == WrongCube) cube ^= 1;
        uint32_t words[4];
        cute::ppu_tsm_ld_swzl_sim<BF16, 16, 16, true>(
            words, memory.data() + cube * 128, 0, 0, 0);
        for (unsigned slot = 0; slot < 8; ++slot) {
          unsigned const native = Native::BLayout{}(lane, slot);
          unsigned const want = tag(plane, slice, row + native / 16,
                                    Eight::column(warp) + native % 16, valid);
          unsigned const got = (words[slot / 2] >> (16 * (slot % 2))) & 65535u;
          bad += got != want;
          ++counts.reads;
        }
      }
  return bad;
}

unsigned suite(Counts& counts, Plant plant = None) {
  unsigned bad = 0;
  for (unsigned plane = 0; plane < 2 - unsigned(plant == MissingPlane); ++plane)
    for (unsigned slice = 0; slice < Dim / ValueTile; ++slice)
      for (unsigned valid = 1; valid <= Chunk; ++valid)
        bad += check(counts, valid, slice, plane, plant);
  // Fixed coverage denominator, independent of any shortened loop above.
  bad += counts.writes != 1048576 || counts.reads != 4194304;
  return bad;
}

int main() {
  Counts counts;
  if (suite(counts)) throw std::runtime_error("eight-warp B writer/native reader mismatch");
  for (Plant plant : {StaleFourWarpBase, StaleValueLayout, WrongSwizzle, WrongCube,
                      MissingLane, MissingWarp, MissingKTile, MissingPlane}) {
    Counts scratch;
    unsigned const bad = suite(scratch, plant);
    if (!bad) throw std::runtime_error("eight-warp B-layout negative escaped");
    std::printf("[warps8 B-layout negative] plant=%d bad=%u OOB=%llu EXPECTED-RED/PASS\n",
                int(plant), bad, (unsigned long long)scratch.out_of_bounds);
  }
  std::printf("[warps8 B-layout] producer_values=%llu reader_values=%llu "
              "planes=2 slices=4 valid=1..64 warps=8 lanes=32 native-C/B+actlize-reader "
              "exact-once=PASS shared=45568 threads=256 layout/precision=PAIRED "
              "device-numerics/BC/performance=NOT_RUN\n",
              (unsigned long long)counts.writes, (unsigned long long)counts.reads);
}
