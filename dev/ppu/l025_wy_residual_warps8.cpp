#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_warps8.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
using Eight = residual_warps8::StateTile;
static_assert(std::is_same_v<residual::Storage, residual_warps8::Storage>);
static_assert(std::is_same_v<residual::Key, residual_warps8::Key>);
static_assert(std::is_same_v<residual::Inverse, residual_warps8::Inverse>);
static_assert(std::is_same_v<residual::Snapshot, residual_warps8::Snapshot>);
static_assert(std::is_same_v<residual::Value, residual_warps8::Value>);
static_assert(residual_warps8::Plan::Threads == 256);

// Execute the actual actlize native reader against tags written by the new
// production owner map. The reader does not apply our offset formula again.
template <bool H>
uint64_t exchange(int plant = 0) {
  using Tile = std::conditional_t<H, residual_warps8::Snapshot, residual_warps8::Value>;
  constexpr int Rows = H ? Dim : Chunk;
  constexpr int Fragments = H ? Eight::KFragments : Eight::ValueFragments;
  std::vector<uint32_t> words(Rows * ValueTile / 2);
  std::vector<unsigned> owners(Rows * ValueTile);
  uint64_t bad = 0, written = 0, read = 0;
  for (int w = 0; w < (plant == 1 ? 4 : Eight::Threads / 32); ++w)
    for (int f = 0; f < Fragments; ++f)
      for (int lane = 0; lane < 32; ++lane)
        for (int s = 0; s < 8; ++s) {
          auto const rc = result_coord(lane, s);
          bad += Native::CLayout{}(lane,s) != rc.row + 16 * rc.col;
          int const warp = plant == 2 ? w % 4 : w;
          int const r = (H ? Eight::k_row(warp,f) : Eight::value_row(warp,f)) + rc.row;
          int const c = Eight::column(warp) + rc.col;
          unsigned at = Tile::offset(r,c);
          if (plant == 3) at ^= 8;
          words.at(at/2) |= uint32_t(r * ValueTile + c + 1) << (16 * (at%2));
          ++owners.at(at);
          ++written;
        }
  for (auto n : owners) bad += n != 1;
  for (int r = 0; r < Rows; r += 16) for (int c = 0; c < ValueTile; c += 16)
    for (int lane = 0; lane < 32 - int(plant == 4); ++lane) {
      threadIdx.x = lane;
      uint32_t frag[4];
      cute::ppu_tsm_ld_swzl_sim<BF16,Rows,ValueTile,true>(frag,words.data(),c,r,0);
      for (int s = 0; s < 8; ++s) {
        int const logical = Native::ALayout{}(lane,s);
        bad += ((frag[s/2] >> (16*(s%2))) & 0xffff) !=
               unsigned((r + logical%16)*ValueTile + c + logical/16 + 1);
        ++read;
      }
    }
  bad += written != unsigned(Rows*ValueTile) || read != unsigned(Rows*ValueTile);
  return bad;
}

template <class Owners>
std::vector<std::vector<int>> trace(int product, int plant = 0) {
  int const rows = product == 2 ? Dim : Chunk;
  int const reduction = product == 0 ? Dim : Chunk;
  int const fragments = product == 2 ? Owners::KFragments : Owners::ValueFragments;
  std::vector<std::vector<int>> out(rows * Dim);
  for (int slice = 0; slice < 4 - int(plant == 7); ++slice)
    for (int w = 0; w < Owners::Threads/32; ++w)
      for (int k = 0; k < reduction - 16*int(plant == 5); k += 16)
        for (int f = 0; f < fragments; ++f)
          for (int lane = 0; lane < 32; ++lane) for (int s = 0; s < 8; ++s) {
            int const native = Native::CLayout{}(lane,s);
            int const r = (product == 2 ? Owners::k_row(w,f) : Owners::value_row(w,f)) + native%16;
            int const c = slice*ValueTile + Owners::column(w) + native/16;
            out.at(r*Dim+c).push_back(plant == 6 ? reduction-16-k : k);
          }
  return out;
}

uint64_t coverage(int plant = 0) {
  uint64_t bad = 0, cells = 0;
  for (int product = 0; product < 3; ++product) {
    auto const old = trace<StateTile>(product);
    auto const now = trace<Eight>(product,plant);
    int const reduction = product == 0 ? Dim : Chunk;
    std::vector<int> expected;
    for (int k = 0; k < reduction; k += 16) expected.push_back(k);
    bad += old != now;
    for (auto const& row : now) { bad += row != expected; cells += row.size(); }
  }
  return bad + (cells != 163840); // Fixed full128-column, three-product denominator.
}

// The shipping global publishers use StateVectorPlan, not the MMA owners.
// Both must change together when a CTA has256 threads. Reconstruct all values
// through the actual contiguous16B shared vectors and independent row-major tags.
template <unsigned Rows, unsigned Bytes>
uint64_t publication(int plant = 0) {
  using Plan = StateVectorPlan<Rows,ValueTile,Bytes,residual_warps8::Plan::Threads>;
  using Tile = aiu::Tile<Rows,ValueTile>;
  std::vector<unsigned> shared(Rows*ValueTile), output(Rows*ValueTile), counts(Rows*ValueTile);
  for (unsigned r=0;r<Rows;++r) for (unsigned c=0;c<ValueTile;++c) {
    unsigned const at = Bytes == 2 ? Tile::offset(r,c) : r*ValueTile+c;
    shared.at(at)=r*ValueTile+c+1;
  }
  uint64_t bad=0, vectors=0;
  for (unsigned tid=0;tid<residual_warps8::Plan::Threads;++tid)
    for (unsigned iteration=0;iteration<Plan::Iterations;++iteration) {
      unsigned const i=plant == 8 ? tid+iteration*128 : Plan::vector(tid,iteration);
      unsigned const r=Plan::row(i),c=Plan::col(i);
      unsigned const at=Bytes == 2 ? Tile::offset(r,c) : r*ValueTile+c;
      for (unsigned x=0;x<Plan::Width;++x) {
        output.at(r*ValueTile+c+x)=shared.at(at+x);
        ++counts.at(r*ValueTile+c+x);
      }
      ++vectors;
    }
  for (unsigned i=0;i<output.size();++i) bad+=output[i]!=i+1 || counts[i]!=1;
  bad+=vectors!=Rows*ValueTile*Bytes/16;
  return bad;
}

int main() {
  if (exchange<false>() || exchange<true>() || coverage() ||
      publication<Dim,2>() || publication<Chunk,2>() || publication<Dim,4>())
    throw std::runtime_error("eight-warp owner/native-read/K-order/publication mismatch");
  for (int plant=1;plant<=8;++plant) {
    uint64_t const bad=plant<=4 ? exchange<true>(plant) :
                       plant<=7 ? coverage(plant) : publication<Dim,2>(plant);
    if (!bad) throw std::runtime_error("eight-warp negative escaped");
    std::printf("[residual warps8 host negative] plant=%d bad=%llu EXPECTED-RED/PASS\n",
                plant,(unsigned long long)bad);
  }
  // Explicit per-warp source work model. Do not confuse matrix-load counts
  // with unique shared words, bank conflicts or measured transaction bytes.
  constexpr int old_loads=(8*(1+StateTile::ValueFragments)+4*(1+StateTile::ValueFragments)+
                          4*(1+StateTile::KFragments))*StateTile::Threads/32;
  constexpr int new_loads=(8*(1+Eight::ValueFragments)+4*(1+Eight::ValueFragments)+
                          4*(1+Eight::KFragments))*Eight::Threads/32;
  static_assert(old_loads==224 && new_loads==288);
  static_assert(sizeof(residual_warps8::Storage)==45568);
  std::puts("[residual warps8 host] owner_values=6144 native-read=6144 "
            "product_cells=163840/all4-V-slices exact-once+old-K-order "
            "publication_vectors=1792 shared=45568 threads=256 "
            "matrix_loads_per_CTA_chunk=224->288 global_inputs=28672B/CTA/chunk "
            "native-traits+actlize-simulator/PASS device=NOT_RUN");
}
