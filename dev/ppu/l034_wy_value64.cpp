#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_value64.cuh"

using namespace gdn_qsa::wy;
namespace W = gdn_qsa::wy::value64;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
void put(std::vector<uint32_t>& words, unsigned at, unsigned tag) {
  unsigned const shift = (at % 2) * 16;
  words.at(at / 2) = (words.at(at / 2) & ~(65535u << shift)) | (tag << shift);
}
unsigned get(std::vector<uint32_t> const& words, unsigned at) { return (words.at(at / 2) >> ((at % 2) * 16)) & 65535; }

template <unsigned Rows, bool IsSnapshot>
uint64_t check(unsigned plant = 0) {
  using M = W::MicroB<Rows, 64, 256>;
  using G = W::GeometryT<64, 256>;
  using P = StateVectorPlan<64, Rows, 2, G::Threads>;
  std::vector<unsigned> global(2 * Rows * Dim), global_owner(global.size());
  uint64_t bad = 0;
  unsigned const fragments = IsSnapshot ? G::KFragments : G::ValueFragments;
  for (unsigned group = 0; group < 2; ++group)
    for (unsigned slice = 0; slice < 2 - unsigned(plant == 5); ++slice) {
      std::vector<uint32_t> shared(Rows * 64 / 2);
      std::vector<unsigned> owner(Rows * 64);
      auto tag = [&](unsigned r, unsigned v) { return 1 + group * Rows * Dim + r * Dim + slice * 64 + v; };
      for (unsigned warp = 0; warp < G::Threads / 32 - unsigned(plant == 1); ++warp)
        for (unsigned f = 0; f < fragments; ++f)
          for (unsigned lane = 0; lane < 32; ++lane)
            for (unsigned s = 0; s < 8; ++s) {
              unsigned const rc = Native::CLayout{}(lane, s);
              unsigned const r = (IsSnapshot ? G::k_row(warp, f) : G::value_row(warp, f)) + rc % 16;
              unsigned const v = G::column(warp) + rc / 16;
              unsigned const base = M::producer_base(warp, lane);
              unsigned at = M::producer_offset(base, f, s);
              if (plant == 2) at ^= 8;
              if (plant == 3) at = base + f * 512 + (s % 4) * 64 + ((s / 4) ^ (s % 2)) * 8;
              bad += at != M::offset(r, v);
              put(shared, at, tag(r, v));
              ++owner.at(at);
            }
      for (auto n : owner) bad += n != 1;
      if (plant == 6) for (auto& x : shared) x = 0xdeadbeef;  // 模拟union过早覆写。
      for (unsigned r = 0; r < Rows; r += 16)
        for (unsigned v = 0; v < 64; v += 16)
          for (unsigned lane = 0; lane < 32; ++lane) {
            threadIdx.x = lane;
            uint32_t words[4];
            cute::ppu_tsm_ld_swzl_sim<BF16, 16, 16, true>(words,
                shared.data() + M::cube(r, v) * 256 / 2, 0, 0, 0);
            for (unsigned s = 0; s < 8; ++s) {
              unsigned const rc = Native::BLayout{}(lane, s);
              bad += ((words[s / 2] >> ((s % 2) * 16)) & 65535) != tag(r + rc / 16, v + rc % 16);
            }
          }
      for (unsigned tid = 0; tid < G::Threads; ++tid)
        for (unsigned it = 0; it < P::Iterations; ++it) {
          unsigned const i = P::vector(tid, it), v = P::row(i), r = P::col(i);
          unsigned const base = M::offset(r, v);
          bad += base % 8 != 0;
          for (unsigned x = 0; x < 8; ++x) {
            bad += M::offset(r + x, v) != base + x;
            unsigned const gv = slice * 64 + v;
            auto const actual = IsSnapshot ? W::SnapshotT<64,256>::workspace_offset(group, r + x, gv)
                                           : W::PublishedValueT<64,256>::workspace_offset(group, r + x, gv);
            unsigned const expected = group * Rows * Dim + gv * Rows + r + x;
            bad += actual != int64_t(expected);
            unsigned const at = plant == 4 ? group * Rows * Dim + gv * (Rows / 2) + r + x : expected;
            global.at(at) = get(shared, base + x);
            ++global_owner.at(at);
          }
        }
    }
  for (unsigned group = 0; group < 2; ++group)
    for (unsigned r = 0; r < Rows; ++r)
      for (unsigned v = 0; v < Dim; ++v) {
        unsigned const at = group * Rows * Dim + v * Rows + r;
        bad += global_owner.at(at) != 1 || global.at(at) != 1 + group * Rows * Dim + r * Dim + v;
      }
  return bad;
}

int main() {
  if (check<128, true>() || check<64, false>())
    throw std::runtime_error("V64/256原生C/B/发布门失败");
  for (unsigned plant = 1; plant <= 6; ++plant) {
    uint64_t const bad = check<128, true>(plant) + check<64, false>(plant);
    if (!bad) throw std::runtime_error("V64/256负控逃逸");
    std::printf("[V64W8 negative] plant=%u bad=%llu EXPECTED-RED/PASS\n", plant, (unsigned long long)bad);
  }
  std::puts("[V64W8 layout] native-C/B+real-SWZL two-groups/two-slices exact-once PASS");
}
