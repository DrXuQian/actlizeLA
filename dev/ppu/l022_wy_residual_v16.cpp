#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_v16.cuh"
#include "gdn_qsa/ppu/wy_split_prepare.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;

// Register-produced residual/snapshot: pin exact owners against REAL CLayout,
// then execute the actlize SWZL simulator, independent of Tile::offset.
template <bool H>
uint64_t exchange(int plant = 0) {
  using Tile = std::conditional_t<H, residual_v16::Snapshot, residual_v16::Value>;
  constexpr int Rows = H ? Dim : Chunk;
  constexpr int Fragments = H ? residual_v16::StateTile::KFragments : residual_v16::StateTile::ValueFragments;
  std::vector<uint32_t> words(Rows * residual_v16::ValueTile / 2);
  std::vector<int> owners(Rows * residual_v16::ValueTile);
  uint64_t bad = 0, count = 0;
  for (int w = 0; w < 4 - int(plant == 1); ++w)
    for (int f = 0; f < Fragments; ++f)
      for (int lane = 0; lane < 32; ++lane) for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        bad += Native::CLayout{}(lane,s) != rc.row + 16 * rc.col;
        int const r = (H ? residual_v16::StateTile::k_row(w,f) : residual_v16::StateTile::value_row(w,f)) + rc.row;
        int const c = residual_v16::StateTile::column(w) + rc.col;
        unsigned at = Tile::offset(r,c);
        // CubeWidth16 occupies exactly the admitted width32 cube's first
        // physical slice. Transposed native loads use the same CUBE_H and
        // coordinates here; the SDK host simulator itself is non-transposed.
        bad += at != aiu::Tile<Rows,32>::offset(r,c);
        if (plant == 2) at ^= 8;
        words.at(at/2) |= uint32_t(r * residual_v16::ValueTile + c + 1) << (16 * (at%2));
        ++owners.at(at);
      }
  for (auto n : owners) bad += n != 1;
  for (int r = 0; r < Rows; r += 16) for (int c = 0; c < residual_v16::ValueTile; c += 16)
    for (int lane = 0; lane < 32; ++lane) {
      threadIdx.x = lane;
      uint32_t frag[4];
      cute::ppu_tsm_ld_swzl_sim<BF16,Rows,residual_v16::ValueTile,true>(frag,words.data(),c,r,0);
      for (int s = 0; s < 8; ++s) {
        int const logical = Native::ALayout{}(lane,s);
        bad += ((frag[s/2] >> (16*(s%2))) & 0xffff) !=
               unsigned((r + logical%16)*residual_v16::ValueTile + c + logical/16 + 1);
        ++count;
      }
    }
  bad += count != unsigned(Rows*residual_v16::ValueTile);
  return bad;
}

uint64_t coverage(int plant = 0) {
  uint64_t bad = 0, cells = 0;
  // KH, P@R and K^T scaledV: every output receives each reduction atom once,
  // in ascending K. Include all8 V slices and exactly all128 output columns.
  for (int product = 0; product < 3; ++product) {
    int const rows = product == 2 ? Dim : Chunk;
    int const reduction = product == 0 ? Dim : Chunk;
    int const fragments = product == 2 ? residual_v16::StateTile::KFragments : residual_v16::StateTile::ValueFragments;
    std::vector<std::vector<int>> traces(rows * Dim);
    for (int slice = 0; slice < 8 - int(plant == 3); ++slice)
      for (int warp = 0; warp < 4; ++warp)
        for (int k = 0; k < reduction; k += 16)
          for (int f = 0; f < fragments; ++f)
            for (int lane = 0; lane < 32; ++lane) for (int s = 0; s < 8; ++s) {
              int const native = Native::CLayout{}(lane,s);
              int const r = (product == 2 ? residual_v16::StateTile::k_row(warp,f) : residual_v16::StateTile::value_row(warp,f)) + native%16;
              int const c = slice*residual_v16::ValueTile + residual_v16::StateTile::column(warp) + native/16;
              traces.at(r*Dim+c).push_back(plant == 4 ? 0 : k);
            }
    std::vector<int> expected;
    for (int k = 0; k < reduction; k += 16) expected.push_back(k);
    for (auto const& trace : traces) { bad += trace != expected; cells += trace.size(); }
  }
  bad += cells != 163840; // 8192*8 +8192*4 +16384*4 (atomic K16 units)
  return bad;
}

uint64_t workspace(int plant = 0) {
  uint64_t bad = 0, count = 0;
  for (int seq = 1; seq <= 4096; ++seq) {
    Shape const shape{2,seq,2,4};
    for (int b = 0; b < 2; ++b) for (int h = 0; h < 4; ++h)
      for (int ct = 0; ct < shape.chunks(); ++ct) {
        int64_t const group = shape.group(b,h,ct);
        int64_t const write = split_prepare::Plan::inverse_base(group);
        int64_t const read = plant == 5 ? tile_offset(group) : residual_v16::Plan::inverse_base(group);
        bad += read != write || read + Chunk*Chunk > shape.groups()*Dim*Dim;
        int const valid = residual_v16::Plan::valid(ct,seq) + int(plant == 6);
        bad += valid != std::min(Chunk,seq - ct*Chunk);
        ++count;
      }
  }
  bad += count != 1064960;
  return bad;
}

int main() {
  if (exchange<false>() || exchange<true>() || coverage() || workspace())
    throw std::runtime_error("residual ownership/native-map/coverage/workspace mismatch");
  for (int plant = 1; plant <= 6; ++plant) {
    uint64_t const bad = plant <= 2 ? exchange<false>(plant) : plant <= 4 ? coverage(plant) : workspace(plant);
    if (!bad) throw std::runtime_error("residual negative escaped");
    std::printf("[residual V16 host negative] plant=%d bad=%llu EXPECTED-RED/PASS\n",plant,(unsigned long long)bad);
  }
  std::puts("[residual V16 host] shared=35328 threads=128 owner_values=3072 product_cells=163840 "
            "group_tail_cases=1064960 native-traits+actlize-simulator/PASS device=NOT_RUN");
}
