// Exhaustive host map checks; native AIU semantics remain a DEVICE test.
#include "csrc/backends/ppu15/layout.cuh"
#include <array>
#include <cstdio>
#include <stdexcept>
using namespace actlize_la::ppu15;
static void require(bool ok, char const* why) { if (!ok) throw std::runtime_error(why); }

template <int Rows, int Cols>
int check_tile() {
  using L = TileLayout<Rows, Cols>;
  std::array<int, Rows * Cols> seen{};
  for (int r = 0; r < Rows; ++r) for (int c = 0; c < Cols; ++c) {
    int const at = L::offset(r,c);
    require(at >= 0 && at < Rows * Cols, "shared offset out of range");
    ++seen[at];
    // Independent bit-position equation, not a second CuTe layout.
    int const local = r * L::Width + c % L::Width;
    int const mask = L::Width == 64 ? 7 : 3;
    int const expected = (c / L::Width) * Rows * L::Width +
                          (local ^ (((local >> 6) & mask) << 3));
    require(at == expected, "SW128/SW64 contract mismatch");
    if (c % 8 == 0) {
      require(at % 8 == 0, "16-byte publication alignment");
      for (int x = 0; x < 8; ++x) require(L::offset(r,c+x) == at+x, "vector crosses a permutation");
    }
  }
  for (int n : seen) require(n == 1, "shared coverage not exact-once");
  return Rows * Cols;
}
int main(int argc, char** argv) {
  try {
    bool const plant = argc > 1;
    std::array<int,256> result{};
    int old_c_differences = 0;
    for (int lane = 0; lane < 32; ++lane) for (int s = 0; s < 8; ++s) {
      auto rc = result_coord(lane,s);
      if (plant) rc = {lane/4+(s/4)*8, lane%4+(s%4)*4};
      require(rc.row == lane/4+((s/2)%2)*8 && rc.col == (lane%4)*2+s%2+(s/4)*8,
              "PPU1.0 result ABI planted in tc02 consumer");
      require(rc.row == lane/4 + 8*row_half(s), "row gate cache mismatch");
      ++result[rc.row*16+rc.col];
      old_c_differences += rc.row != lane/4+(s/4)*8 || rc.col != lane%4+(s%4)*4;
      auto a = coordinate<typename cute::MMA_Traits<Bf16Atom>::ALayout,16>(lane,s);
      auto b = coordinate<typename cute::MMA_Traits<Bf16Atom>::BLayout,16>(lane,s);
      require(a.row == rc.row && a.col == rc.col, "tc02 A/C mismatch");
      require(b.row == lane/4+(s/4)*8 && b.col == (lane%4)*2+s%2+((s/2)%2)*8,
              "tc02 B map mismatch");
    }
    for (int n : result) require(n == 1, "result ownership not exact-once");
    for (int lane = 0; lane < 32; ++lane) for (int s = 0; s < 4; ++s) {
      auto a = coordinate<typename cute::MMA_Traits<Tf32Atom>::ALayout,16>(lane,s);
      auto b = coordinate<typename cute::MMA_Traits<Tf32Atom>::BLayout,16>(lane,s);
      require(a.row == lane/4+(s%2)*8 && a.col == lane%4+(s/2)*4, "TF32 A map");
      require(b.row == lane/4+(s/2)*8 && b.col == lane%4+(s%2)*4, "TF32 B map");
    }
    std::array<int,128*32> h{};
    std::array<int,64*32> value{};
    std::array<int,64*64> out{};
    for (int tid = 0; tid < 256; ++tid) for (int s = 0; s < 8; ++s) {
      int const warp = tid/32;
      auto const rc = result_coord(tid%32,s);
      for (int f = 0; f < StateTile::KFragments; ++f)
        ++h[(StateTile::k_row(warp,f)+rc.row)*32+StateTile::col(warp)+rc.col];
      ++value[(StateTile::value_row(warp)+rc.row)*32+StateTile::col(warp)+rc.col];
      for (int f = 0; f < OutputTile::Fragments; ++f)
        ++out[(OutputTile::row(warp)+rc.row)*64+OutputTile::col(warp,f)+rc.col];
    }
    for (auto n : h) require(n == 1, "state duplicate/missing owner");
    for (auto n : value) require(n == 1, "V duplicate/missing owner");
    for (auto n : out) require(n == 1, "output duplicate/missing owner");
    int cells = check_tile<64,128>() + check_tile<64,64>() + check_tile<64,32>() +
                check_tile<32,128>() + check_tile<32,64>();
    std::printf("[ppu15 maps] PASS shared_cells=%d exact-once; old-C-differences=%d/256; device=NOT_RUN\n", cells, old_c_differences);
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr,"[ppu15 maps] FAIL: %s\n",e.what());
    return 1;
  }
}
