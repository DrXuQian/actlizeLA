// L214: exhaustive host authority for the post-cumsum Triton-aligned DAG.
// It proves stage ownership and the exact A/W/U/H/Vnew workspace denominator
// for Qwen3.5-35B-A3B T2048.  It deliberately does not model device math.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_types.hpp"

namespace {
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
using Chunk = cutlass::linear_attention::PpuChunkedGdnPrepareScheduler<Traits>;
using H = cutlass::linear_attention::PpuChunkedGdnScheduler<Traits, 64>;
using O = cutlass::linear_attention::PpuChunkedGdnValueChunkScheduler<Traits, 64>;

struct Counts { int holes = 0, duplicates = 0, out_of_range = 0; };
Counts counts(std::vector<int> const& visits, int out_of_range = 0) {
  Counts c{};
  c.out_of_range = out_of_range;
  for (int n : visits) {
    c.holes += n == 0;
    c.duplicates += n > 1 ? n - 1 : 0;
  }
  return c;
}
}  // namespace

int main() {
  cutlass::linear_attention::PpuChunkedGdnProblem const problem{
      2048, 1, 2048, 16, 32, 128, 128, 64};
  constexpr int kChunks = 32;
  constexpr int kHeads = 32;
  constexpr int kValueTiles = 2;
  constexpr int kChunkCells = kHeads * kChunks;
  constexpr int kHCells = kHeads * kValueTiles;
  constexpr int kOCells = kChunkCells * kValueTiles;
  constexpr std::int64_t kBytesPerChunk = 90112;
  constexpr std::int64_t kWorkspaceBytes =
      std::int64_t(kChunkCells) * kBytesPerChunk;

  std::vector<int> kkt(kChunkCells), wu(kChunkCells), h(kHCells), o(kOCells);
  int mapping_bad = 0;
  for (int block = 0; block < Chunk::grid_size(problem); ++block) {
    auto const work = Chunk::work(block, problem);
    int const index = work.head.v_head_idx * kChunks + work.chunk_idx;
    mapping_bad += !work.valid || index != block;
    if (index >= 0 && index < kChunkCells) ++kkt[index], ++wu[index];
  }
  int h_chunk_visits = 0;
  for (int block = 0; block < H::grid_size(problem); ++block) {
    auto const work = H::work(block, problem);
    int const index = work.v_head_idx * kValueTiles + work.value_tile_idx;
    mapping_bad += !work.valid || index != block || work.chunk_count != kChunks;
    if (index >= 0 && index < kHCells) ++h[index];
    h_chunk_visits += work.chunk_count;
  }
  for (int block = 0; block < O::grid_size(problem); ++block) {
    auto const work = O::work(block, problem);
    int const index =
        (work.head.v_head_idx * kChunks + work.chunk_idx) * kValueTiles +
        work.head.value_tile_idx;
    mapping_bad += !work.valid || index != block;
    if (index >= 0 && index < kOCells) ++o[index];
  }

  Counts const ck = counts(kkt), cw = counts(wu), ch = counts(h), co = counts(o);

  // Plant 1: split W/U by BV64 like v3. This doubles a stage that Triton owns
  // once per head/chunk.
  int const split_wu_grid = O::grid_size(problem);
  bool const split_wu_red = split_wu_grid != kChunkCells;
  // Plant 2: multiply H by chunks. This destroys the one serial chain owner.
  int const chunk_parallel_h_grid = H::grid_size(problem) * kChunks;
  bool const chunk_h_red = chunk_parallel_h_grid != kHCells;
  // Plant 3: omit one workspace cell from the declared denominator.
  std::vector<int> short_workspace(kChunkCells - 1);
  int short_oob = 0;
  for (int block = 0; block < Chunk::grid_size(problem); ++block) {
    auto const work = Chunk::work(block, problem);
    int const index = work.head.v_head_idx * kChunks + work.chunk_idx;
    if (index >= int(short_workspace.size())) ++short_oob;
    else ++short_workspace[index];
  }
  Counts const short_denominator = counts(short_workspace, short_oob);
  // Plant 4: reintroduce the causal CxC BF16 P seam removed by on-the-fly O.
  constexpr std::int64_t kMaterializedPBytesPerChunk =
      kBytesPerChunk + 64 * 64 * 2;
  bool const p_seam_red = kMaterializedPBytesPerChunk != kBytesPerChunk;

  bool const positive =
      Chunk::grid_size(problem) == kChunkCells &&
      H::grid_size(problem) == kHCells && O::grid_size(problem) == kOCells &&
      ck.holes + ck.duplicates + cw.holes + cw.duplicates +
          ch.holes + ch.duplicates + co.holes + co.duplicates == 0 &&
      h_chunk_visits == kOCells && mapping_bad == 0 &&
      kWorkspaceBytes == 88ll * 1024 * 1024;
  bool const negatives = split_wu_red && chunk_h_red && p_seam_red &&
                         short_denominator.out_of_range == 1;
  std::printf(
      "[L214 triton alignment] shape=B1,T2048,H16,HV32,K128,V128,C64 "
      "grid=KKT:%d/WU:%d/H:%d/O:%d workspace=%lldx%d=%lld "
      "coverage_bad=%d chain_cells=%d map_bad=%d\n",
      Chunk::grid_size(problem), Chunk::grid_size(problem),
      H::grid_size(problem), O::grid_size(problem),
      static_cast<long long>(kBytesPerChunk), kChunkCells,
      static_cast<long long>(kWorkspaceBytes),
      ck.holes + ck.duplicates + cw.holes + cw.duplicates +
          ch.holes + ch.duplicates + co.holes + co.duplicates,
      h_chunk_visits, mapping_bad);
  std::printf(
      "[L214 plants] split-WU-grid=%d serial-H-times-chunks=%d "
      "short-denominator-oob=%d materialized-P-bytes/chunk=%lld %s\n",
      split_wu_grid, chunk_parallel_h_grid, short_denominator.out_of_range,
      static_cast<long long>(kMaterializedPBytesPerChunk),
      negatives ? "EXPECTED_RED/PASS" : "FAIL");
  std::printf("[L214] %s: post-cumsum stage ownership exhaustive\n",
              positive && negatives ? "PASS" : "FAIL");
  return positive && negatives ? 0 : 1;
}
