#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <algorithm>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_state_pipeline.cuh"

using namespace gdn_qsa::wy;
using state_pipeline::Plan;

// Partial-order proof, not a simulated floating-point GPU. Each warp may run
// arbitrarily far ahead until a CTA barrier. AIU writes occupy an interval
// [issue,completion]; a barrier alone does not complete asynchronous writes.
// Disjoint per-warp stores are represented separately. Conflicting intervals
// MUST be ordered by the real waits/barriers, regardless of warp interleaving.
enum Plane { W, K, U, H, Scaled, G };
struct Access { Plane plane; unsigned owners; bool write; int start, end; };
struct Graph {
  std::vector<std::vector<uint64_t>> ancestors;
  std::array<int, 4> last{{-1,-1,-1,-1}};
  std::vector<int> pending;
  std::vector<Access> accesses;
  int node(std::vector<int> const& parents) {
    size_t const id = ancestors.size();
    std::vector<uint64_t> bits((id + 64) / 64);
    for (int parent : parents) if (parent >= 0) {
      auto const& other = ancestors.at(size_t(parent));
      for (size_t i = 0; i < other.size(); ++i) bits[i] |= other[i];
      bits[size_t(parent) / 64] |= uint64_t(1) << (unsigned(parent) % 64);
    }
    ancestors.push_back(std::move(bits));
    return int(id);
  }
  int event(int warp, std::vector<int> deps = {}) {
    deps.push_back(last.at(size_t(warp)));
    return last.at(size_t(warp)) = node(deps);
  }
  void barrier() {
    int const join = node({last[0],last[1],last[2],last[3]});
    last.fill(join);
  }
  void copy(Plane plane) {
    int const issue = event(0);
    int const completion = node({issue});  // no implicit ordering across AIU operations
    pending.push_back(completion);
    accesses.push_back({plane,15,true,issue,completion});
  }
  void wait() { event(0, pending); pending.clear(); }
  void access(int warp, Plane plane, bool write, unsigned owners = 15) {
    int const at = event(warp);
    accesses.push_back({plane,owners,write,at,at});
  }
  bool before(int a, int b) const {
    if (a == b) return true;
    auto const& bits = ancestors.at(size_t(b));
    return size_t(a) / 64 < bits.size() && (bits[size_t(a) / 64] >> (unsigned(a) % 64) & 1);
  }
  uint64_t conflicts() const {
    uint64_t bad = 0;
    for (size_t i = 0; i < accesses.size(); ++i) for (size_t j = i + 1; j < accesses.size(); ++j) {
      auto const& a = accesses[i]; auto const& b = accesses[j];
      if (a.plane != b.plane || !(a.owners & b.owners) || (!a.write && !b.write)) continue;
      bad += !before(a.end,b.start) && !before(b.end,a.start);
    }
    return bad;
  }
};

struct Counts { uint64_t shapes = 0, chunks = 0, cells = 0, accesses = 0, pairs = 0; };

uint64_t lifetime(int chunks, bool final, Counts& count, int plant = 0) {
  Graph g;
  g.copy(W);  // prologue, one group: commit is bound to source/native by the companion gate
  for (int ct = 0; ct < chunks; ++ct) {
    for (int warp = 0; warp < 4; ++warp) g.access(warp,H,true,1u << warp);
    if (plant != 1) g.wait();                         // W completion
    if (plant != 2) g.barrier();                      // W/snapshot ready; old K/U retired
    for (int warp = 0; warp < 4; ++warp) g.access(warp,H,false);
    g.copy(K); g.copy(U);
    for (int warp = 0; warp < 2; ++warp) g.access(warp,G,true,1u << warp);
    for (int warp = 0; warp < 4; ++warp) {
      g.access(warp,W,false); g.access(warp,H,false); // W@H
    }
    if (plant != 3) g.wait();                         // K/U completion
    if (plant != 4) g.barrier();                      // W readers retired
    if (Plan::next(ct,chunks) >= 0 || plant == 6) g.copy(W);
    for (int warp = 0; warp < 4; ++warp) {
      g.access(warp,G,false,3);
      g.access(warp,U,false,1u << warp);
      g.access(warp,U,true,1u << warp);
      g.access(warp,Scaled,true,1u << warp);
    }
    if (plant != 5) g.barrier();                      // new/scaled V ready, NOT an AIU wait
    for (int warp = 0; warp < 4; ++warp) {
      g.access(warp,U,false);                        // vector publication
      g.access(warp,K,false); g.access(warp,Scaled,false);  // K^T scaledV
    }
    // There is deliberately no end-of-iteration CTA barrier.
  }
  if (final) {
    for (int warp = 0; warp < 4; ++warp) g.access(warp,W,true,1u << warp); // final_h alias
    if (plant != 7) g.barrier();
    for (int warp = 0; warp < 4; ++warp) g.access(warp,W,false);
  }
  count.accesses += g.accesses.size();
  count.pairs += g.accesses.size() * (g.accesses.size() - 1) / 2;
  // Even output-only must not return while an extra last-iteration W writes.
  return g.conflicts() + !g.pending.empty();
}

uint64_t coordinates(Counts& count, int plant = 0) {
  uint64_t bad = 0;
  // Exhaustive domain includes every tail, every1..128 chunk count, ragged
  // final chunks, and multiple sequence/head groups (no random sampling).
  for (int sequence = 1; sequence <= 8192; ++sequence) for (int bh = 0; bh < 8; ++bh) {
    Shape const shape{2,sequence,2,4};
    int const chunks = shape.chunks(), b = bh / 4, h = bh % 4;
    std::vector<int> w_visits(size_t(chunks),0);
    ++w_visits[0];  // prologue
    for (int ct = 0; ct < chunks; ++ct) {
      int const valid = Plan::valid(ct,sequence) + int(plant == 8);
      for (int row = 0; row < Chunk; ++row) {
        bad += (row < valid) != (ct * Chunk + row < sequence);
        ++count.cells;
      }
      int const next = Plan::next(ct,chunks);
      bad += (next >= 0) != (ct + 1 < chunks);
      if (next >= 0) {
        ++w_visits.at(size_t(next));
        auto const group = shape.group(b,h,plant == 9 ? ct : next);
        bad += group != int64_t(bh) * chunks + ct + 1;
        bad += tile_offset(group) != (int64_t(bh) * chunks + ct + 1) * 8192;
      }
      ++count.chunks;
    }
    for (int n : w_visits) bad += n != 1;
    ++count.shapes;
  }
  bad += count.shapes != 65536 || count.chunks != 4227072 || count.cells != 270532608;
  return bad;
}

int main() {
  Counts counts;
  if (coordinates(counts)) throw std::runtime_error("state pipeline chunk/tail/group mismatch");
  for (int chunks = 1; chunks <= 33; ++chunks) for (bool final : {false,true})
    if (lifetime(chunks,final,counts)) throw std::runtime_error("state pipeline unordered conflicting access");
  for (int plant = 1; plant <= 9; ++plant) {
    Counts ignored;
    uint64_t const bad = plant <= 7 ? lifetime(3,true,ignored,plant) : coordinates(ignored,plant);
    if (!bad) throw std::runtime_error("state pipeline negative escaped");
    std::printf("[WY pipeline negative] plant=%d bad=%llu EXPECTED-RED/PASS\n",plant,(unsigned long long)bad);
  }
  if (StatePipelineDelivery != 62960 || !valid_delivery(62960) || !valid_delivery(30192))
    throw std::runtime_error("pipeline selector not admitted");
  for (unsigned mask = 32768; mask < 65536; ++mask)
    if (valid_delivery(mask) != (mask == 62960)) throw std::runtime_error("pipeline selector fail-open");
  std::printf("[WY state pipeline] shapes=%llu chunks=%llu tail_cells=%llu lifetime_schedules=66 "
              "accesses=%llu interval_pairs=%llu shared=%u threads=%d conflicts=0 "
              "PASS device_execution=NOT_RUN\n",(unsigned long long)counts.shapes,
              (unsigned long long)counts.chunks,(unsigned long long)counts.cells,
              (unsigned long long)counts.accesses,(unsigned long long)counts.pairs,Plan::SharedBytes,Plan::Threads);
}
