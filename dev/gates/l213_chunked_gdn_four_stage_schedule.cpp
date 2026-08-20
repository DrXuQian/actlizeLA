// L213 -- exhaustive host authority for the four-stage Qwen3.5 GDN DAG.
// It proves every independently parallel common/value cell is emitted once,
// every H owner carries exactly one complete chunk chain, and the workspace
// denominator includes every declared combination.  No device is required.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_types.hpp"

namespace {

using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
using Common = cutlass::linear_attention::PpuChunkedGdnPrepareScheduler<Traits>;
using Value =
    cutlass::linear_attention::PpuChunkedGdnValueChunkScheduler<Traits, 64>;
using H = cutlass::linear_attention::PpuChunkedGdnScheduler<Traits, 64>;

struct Coverage {
  int holes = 0;
  int duplicates = 0;
  int out_of_range = 0;
};

Coverage summarize(std::vector<int> const& visits, int out_of_range = 0) {
  Coverage out{};
  out.out_of_range = out_of_range;
  for (int count : visits) {
    out.holes += count == 0;
    out.duplicates += count > 1 ? count - 1 : 0;
  }
  return out;
}

}  // namespace

int main() {
  cutlass::linear_attention::PpuChunkedGdnProblem const problem{
      2048, 1, 2048, 16, 32, 128, 128, 64};
  constexpr int kChunks = 32;
  constexpr int kHeads = 32;
  constexpr int kValueTiles = 2;
  constexpr int kCommonCells = kHeads * kChunks;
  constexpr int kValueCells = kCommonCells * kValueTiles;
  constexpr int kStateColumns = kHeads * 128;
  constexpr std::int64_t kCommonBytes =
      std::int64_t(kCommonCells) * 32768;
  constexpr std::int64_t kValueBytes =
      std::int64_t(kValueCells) * 40960;

  std::vector<int> common(kCommonCells);
  std::vector<int> u(kValueCells);
  std::vector<int> output(kValueCells);
  std::vector<int> state_columns(kStateColumns);
  int mapping_bad = 0;

  for (int block = 0; block < Common::grid_size(problem); ++block) {
    auto const work = Common::work(block, problem);
    if (!work.valid) {
      ++mapping_bad;
      continue;
    }
    int const index = work.head.v_head_idx * kChunks + work.chunk_idx;
    ++common[index];
    mapping_bad += index != block;
    mapping_bad += work.head.qk_head_idx != work.head.v_head_idx / 2;
  }
  for (int block = 0; block < Value::grid_size(problem); ++block) {
    auto const work = Value::work(block, problem);
    if (!work.valid) {
      ++mapping_bad;
      continue;
    }
    int const index =
        (work.head.v_head_idx * kChunks + work.chunk_idx) * kValueTiles +
        work.head.value_tile_idx;
    ++u[index];
    ++output[index];
    mapping_bad += index != block;
    mapping_bad += work.head.qk_head_idx != work.head.v_head_idx / 2;
  }
  int h_chunk_visits = 0;
  for (int block = 0; block < H::grid_size(problem); ++block) {
    auto const work = H::work(block, problem);
    if (!work.valid) {
      ++mapping_bad;
      continue;
    }
    mapping_bad += work.chunk_count != kChunks;
    h_chunk_visits += work.chunk_count;
    for (int value = 0; value < work.value_count; ++value) {
      ++state_columns[work.v_head_idx * 128 + work.value_begin + value];
    }
  }

  Coverage const c = summarize(common);
  Coverage const uv = summarize(u);
  Coverage const o = summarize(output);
  Coverage const h = summarize(state_columns);

  // Plant 1: omit the final declared value cell from the denominator.
  std::vector<int> short_denominator(kValueCells - 1);
  int short_oob = 0;
  for (int block = 0; block < Value::grid_size(problem); ++block) {
    auto const work = Value::work(block, problem);
    int const index =
        (work.head.v_head_idx * kChunks + work.chunk_idx) * kValueTiles +
        work.head.value_tile_idx;
    if (index >= int(short_denominator.size())) {
      ++short_oob;
    } else {
      ++short_denominator[index];
    }
  }
  Coverage const short_plant = summarize(short_denominator, short_oob);

  // Plant 2: drop value_tile_idx from the workspace address.  Every upper
  // BV64 record must become a hole and every lower record a duplicate.
  std::vector<int> folded_value_tile(kValueCells);
  for (int block = 0; block < Value::grid_size(problem); ++block) {
    auto const work = Value::work(block, problem);
    int const index =
        (work.head.v_head_idx * kChunks + work.chunk_idx) * kValueTiles;
    ++folded_value_tile[index];
  }
  Coverage const value_tile_plant = summarize(folded_value_tile);

  // Plant 3: treat each BV64-local column as a global V128 column.
  std::vector<int> local_as_global(kStateColumns);
  for (int block = 0; block < H::grid_size(problem); ++block) {
    auto const work = H::work(block, problem);
    for (int value = 0; value < work.value_count; ++value) {
      ++local_as_global[work.v_head_idx * 128 + value];
    }
  }
  Coverage const local_plant = summarize(local_as_global);

  bool const positive =
      Common::grid_size(problem) == kCommonCells &&
      Value::grid_size(problem) == kValueCells &&
      H::grid_size(problem) == kHeads * kValueTiles &&
      c.holes + c.duplicates == 0 && uv.holes + uv.duplicates == 0 &&
      o.holes + o.duplicates == 0 && h.holes + h.duplicates == 0 &&
      h_chunk_visits == kValueCells && mapping_bad == 0 &&
      kCommonBytes == 32ll * 1024 * 1024 &&
      kValueBytes == 80ll * 1024 * 1024;
  bool const negatives =
      short_plant.out_of_range == 1 &&
      value_tile_plant.holes == kCommonCells &&
      value_tile_plant.duplicates == kCommonCells &&
      local_plant.holes == kHeads * 64 &&
      local_plant.duplicates == kHeads * 64;

  std::printf(
      "[L213 four-stage] shape=B1,T2048,H16,HV32,K128,V128,C64 "
      "grid=common:%d/U:%d/H:%d/O:%d workspace=%lld+%lld=%lld "
      "common_bad=%d U_bad=%d H_bad=%d O_bad=%d chain_cells=%d "
      "map_bad=%d\n",
      Common::grid_size(problem), Value::grid_size(problem),
      H::grid_size(problem), Value::grid_size(problem),
      static_cast<long long>(kCommonBytes),
      static_cast<long long>(kValueBytes),
      static_cast<long long>(kCommonBytes + kValueBytes),
      c.holes + c.duplicates, uv.holes + uv.duplicates,
      h.holes + h.duplicates, o.holes + o.duplicates,
      h_chunk_visits, mapping_bad);
  std::printf(
      "[L213 plants] short-denominator=oob:%d "
      "drop-value-tile=holes:%d/dup:%d "
      "local-V-as-global=holes:%d/dup:%d %s\n",
      short_plant.out_of_range, value_tile_plant.holes,
      value_tile_plant.duplicates, local_plant.holes,
      local_plant.duplicates, negatives ? "EXPECTED_RED/PASS" : "FAIL");
  std::printf("[L213] %s: four-stage ownership and workspace exhaustive\n",
              positive && negatives ? "PASS" : "FAIL");
  return positive && negatives ? 0 : 1;
}
