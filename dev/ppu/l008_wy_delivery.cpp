#include <array>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/wy_contract.hpp"
#include "gdn_qsa/ppu/wy_delivery.cuh"

using namespace gdn_qsa::wy;

// Native C producers -> shared exchange -> vector global transactions. Values
// are unique bit markers, not a tolerance-based numerical comparison.
template <int Rows, int Cols, int Threads, int Bytes>
int check(int valid, int stride, int base, int plant = 0) {
  using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
  using Vectors = VectorTile<Rows, Cols, Bytes>;
  std::vector<int> shared(Rows * Cols, -1), shared_owners(Rows * Cols);
  std::vector<int> output(base + Rows * stride + 16, -1), owners(output.size());
  int bad = 0;
  for (int br = 0; br < Rows; br += 16)
    for (int bc = 0; bc < Cols; bc += 16)
      for (int lane = 0; lane < 32; ++lane)
        for (int slot = 0; slot < 8; ++slot) {
          auto const rc = result_coord(lane, slot);
          bad += Native::CLayout{}(lane, slot) != rc.row + 16 * rc.col;
          int const r = br + rc.row, c = bc + rc.col;
          int const at = Bytes == 2 ? swizzle<Rows, Cols>(r, c) : r * Cols + c;
          shared[at] = r * Cols + c;
          ++shared_owners[at];
        }
  for (int x : shared_owners) bad += x != 1;
  int visited = 0;
  for (int tid = 0; tid < Threads; ++tid)
    for (int i = tid; i < Vectors::Count - int(plant == 2); i += Threads) {
      ++visited;
      int const r = Vectors::row(i), c = Vectors::col(i);
      int const at = Bytes == 2 ? swizzle<Rows, Cols>(r, c) : r * Cols + c;
      bad += (at * Bytes) % 16 != 0 || ((base + r * stride + c) * Bytes) % 16 != 0;
      for (int e = 0; e < Vectors::Width; ++e) {
        int const physical = Bytes == 2 ? swizzle<Rows, Cols>(r, c + e) : r * Cols + c + e;
        bad += physical != at + e;
        if (r < valid || (plant == 3 && r == valid)) {
          int const dst = base + r * stride + c + e;
          output[dst] = shared[at + (e ^ int(plant == 1))];
          ++owners[dst];
        }
      }
    }
  bad += visited != Vectors::Count;
  for (size_t i = 0; i < output.size(); ++i) {
    int const logical = int(i) - base;
    bool const wanted = logical >= 0 && logical / stride < valid && logical % stride < Cols;
    int const expected = wanted ? (logical / stride) * Cols + logical % stride : -1;
    bad += output[i] != expected || owners[i] != int(wanted);
  }
  return bad;
}

template <int Rows, int Cols, int Threads, int Bytes>
int exhaustive() {
  int cases = 0;
  for (int valid = 0; valid <= Rows; ++valid)
    for (int stride : {Cols, 128, 256, 4096})
      for (int base : {0, 16, 32}) {
        if (check<Rows, Cols, Threads, Bytes>(valid, stride, base))
          throw std::runtime_error("native-to-vector ownership or tail failure");
        ++cases;
      }
  for (int plant : {1, 2, 3}) {
    int const bad = check<Rows, Cols, Threads, Bytes>(Rows - 1, 128, 16, plant);
    if (!bad) throw std::runtime_error("delivery negative escaped");
    std::printf("[WY delivery negative] tile=%dx%d bytes=%d plant=%d bad=%d EXPECTED-RED/PASS\n",
        Rows, Cols, Bytes, plant, bad);
  }
  return cases;
}

int main() {
  int cases = exhaustive<16, 16, 32, 2>();  // warp-private prepare/output
  cases += exhaustive<128, 32, 64, 2>();  // full-CTA incoming H
  cases += exhaustive<64, 32, 64, 2>();   // U and Vnew
  cases += exhaustive<128, 32, 64, 4>(); // final FP32 state
  cases += exhaustive<128, 32, 128, 2>(); // four-warp H snapshot
  cases += exhaustive<64, 32, 128, 2>();  // four-warp Vnew
  cases += exhaustive<128, 32, 128, 4>(); // four-warp final state
  cases += exhaustive<64, 64, 128, 2>();  // CTA-wide W/U panels
  cases += exhaustive<64, 64, 256, 2>();  // eight-warp output panels
  std::printf("[WY delivery] cases=%d all-tails/padded-strides/offsets exact-once/PASS device=NOT_RUN\n", cases);
}
