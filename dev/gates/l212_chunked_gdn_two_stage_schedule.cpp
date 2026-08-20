// L212 -- exhaustive host authority for the Qwen3.5-35B-A3B T=2048
// prepare/recurrence decomposition.  No sampling and no device execution.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_types.hpp"

namespace {

using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
using Prepare = cutlass::linear_attention::PpuChunkedGdnPrepareScheduler<Traits>;
using Recurrence = cutlass::linear_attention::PpuChunkedGdnScheduler<Traits, 64>;

struct Coverage {
  int holes = 0;
  int duplicates = 0;
  int out_of_range = 0;
};

Coverage summarize(std::vector<int> const& visits, int out_of_range = 0) {
  Coverage c{};
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
  constexpr int kPrepareCells = 32 * kChunks;
  constexpr int kOutputColumns = 32 * 128;
  constexpr std::int64_t kWorkspaceBytes =
      std::int64_t(kPrepareCells) *
      cutlass::linear_attention::PpuChunkedGdnPreparedChunkBytes<Traits>;

  std::vector<int> prepare(kPrepareCells);
  std::vector<int> recurrence(kOutputColumns);
  int bad_head_map = 0;
  for (int block = 0; block < Prepare::grid_size(problem); ++block) {
    auto const w = Prepare::work(block, problem);
    if (!w.valid) {
      ++bad_head_map;
      continue;
    }
    int const index = w.head.v_head_idx * kChunks + w.chunk_idx;
    ++prepare[index];
    bad_head_map += w.head.qk_head_idx != w.head.v_head_idx / 2;
  }
  for (int block = 0; block < Recurrence::grid_size(problem); ++block) {
    auto const w = Recurrence::work(block, problem);
    if (!w.valid) {
      ++bad_head_map;
      continue;
    }
    bad_head_map += w.qk_head_idx != w.v_head_idx / 2;
    for (int value = 0; value < w.value_count; ++value) {
      ++recurrence[w.v_head_idx * 128 + w.value_begin + value];
    }
  }
  Coverage const p = summarize(prepare);
  Coverage const r = summarize(recurrence);

  // Plant 1: the denominator silently omits one declared combination.
  std::vector<int> short_denominator(kPrepareCells - 1);
  int short_oob = 0;
  for (int block = 0; block < Prepare::grid_size(problem); ++block) {
    auto const w = Prepare::work(block, problem);
    int const index = w.head.v_head_idx * kChunks + w.chunk_idx;
    if (index >= int(short_denominator.size())) {
      ++short_oob;
    } else {
      ++short_denominator[index];
    }
  }
  Coverage const short_plant = summarize(short_denominator, short_oob);

  // Plant 2: fold chunk 31 onto chunk 0.  Every V head must expose one hole
  // and one duplicate; a sampled first/last-head check is insufficient.
  std::vector<int> folded(kPrepareCells);
  for (int block = 0; block < Prepare::grid_size(problem); ++block) {
    auto const w = Prepare::work(block, problem);
    int const chunk = w.chunk_idx == 31 ? 0 : w.chunk_idx;
    ++folded[w.head.v_head_idx * kChunks + chunk];
  }
  Coverage const folded_plant = summarize(folded);

  // Plant 3: use the local BV64 column as a global V128 column.  This is the
  // exact split-V ownership error: lower columns duplicate, upper columns hole.
  std::vector<int> local_as_global(kOutputColumns);
  for (int block = 0; block < Recurrence::grid_size(problem); ++block) {
    auto const w = Recurrence::work(block, problem);
    for (int value = 0; value < w.value_count; ++value) {
      ++local_as_global[w.v_head_idx * 128 + value];
    }
  }
  Coverage const local_plant = summarize(local_as_global);

  bool const positive = Prepare::grid_size(problem) == 1024 &&
                        Recurrence::grid_size(problem) == 64 &&
                        kWorkspaceBytes == 32ll * 1024 * 1024 &&
                        p.holes == 0 && p.duplicates == 0 &&
                        r.holes == 0 && r.duplicates == 0 && bad_head_map == 0;
  bool const negatives = short_plant.out_of_range == 1 &&
                         folded_plant.holes == 32 &&
                         folded_plant.duplicates == 32 &&
                         local_plant.holes == 2048 &&
                         local_plant.duplicates == 2048;
  std::printf(
      "[L212 Qwen3.5-35B-A3B] B=1 T=2048 Hqk=16 Hv=32 K=128 V=128 C=64 "
      "prepare_grid=%d recurrence_grid=%d workspace=%lld "
      "prepare_exact_once=%d recurrence_exact_once=%d head_map_bad=%d\n",
      Prepare::grid_size(problem), Recurrence::grid_size(problem),
      static_cast<long long>(kWorkspaceBytes),
      p.holes + p.duplicates, r.holes + r.duplicates, bad_head_map);
  std::printf(
      "[L212 plants] short-denominator=oob:%d folded-chunk=holes:%d/dup:%d "
      "local-V-as-global=holes:%d/dup:%d %s\n",
      short_plant.out_of_range, folded_plant.holes,
      folded_plant.duplicates, local_plant.holes, local_plant.duplicates,
      negatives ? "EXPECTED_RED/PASS" : "FAIL");
  std::printf("[L212] %s: two-stage schedule/workspace exhaustive\n",
              positive && negatives ? "PASS" : "FAIL");
  return positive && negatives ? 0 : 1;
}
