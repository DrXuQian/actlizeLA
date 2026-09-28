#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <algorithm>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_prefetch.cuh"

using namespace gdn_qsa::wy;
using residual_prefetch::Plan;

// Partial-order proof, not a simulated floating-point GPU. Each warp may run
// arbitrarily far ahead until a CTA barrier. AIU writes occupy an interval
// [issue,completion]; a barrier alone does not complete asynchronous writes.
// Disjoint per-warp stores are represented separately. Conflicting intervals
// MUST be ordered by the real waits/barriers, regardless of warp interleaving.
enum Plane { P, K, V, H, R, Scaled, G, Beta };
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


uint64_t lifetime(int chunks, bool final, int plant = 0) {
  Graph g;
  g.copy(P);
  for (int ct = 0; ct < chunks; ++ct) {
    g.copy(K); g.copy(V);
    for (int w = 0; w < 2; ++w) {
      g.access(w,G,true,1u << w); g.access(w,Beta,true,1u << w);
    }
    for (int w = 0; w < 4; ++w) g.access(w,H,true,1u << w);
    if (plant != 1) g.wait();
    if (plant != 2) g.barrier();             // INPUTS_READY
    for (int w = 0; w < 4; ++w) {
      g.access(w,H,false);                   // snapshot publication and KH
      g.access(w,K,false);
      g.access(w,V,false); g.access(w,G,false); g.access(w,Beta,false);
      g.access(w,R,true,1u << w);
    }
    if (plant != 3) g.barrier();             // RESIDUAL_READY
    for (int w = 0; w < 4; ++w) {
      g.access(w,P,false); g.access(w,R,false);
      g.access(w,V,true,1u << w); g.access(w,Scaled,true,1u << w);
    }
    if (plant != 4) g.barrier();             // VALUES_READY retires P readers
    if (Plan::next(ct,chunks) >= 0) g.copy(P);
    for (int w = 0; w < 4; ++w) {
      g.access(w,V,false);
      g.access(w,K,false); g.access(w,Scaled,false); // H update overlaps next P
    }
    if (plant != 5) g.barrier();             // RETIRE before next K/V writers
  }
  if (final) {
    for (int w = 0; w < 4; ++w) g.access(w,K,true,1u << w); // K/final_h union
    if (plant != 6) g.barrier();
    for (int w = 0; w < 4; ++w) g.access(w,K,false);
  }
  return g.conflicts();
}

uint64_t coordinates(int plant = 0) {
  uint64_t bad=0, shapes=0, chunks_seen=0, rows=0;
  for (int sequence=1; sequence<=4096; ++sequence) for (int bh=0; bh<8; ++bh) {
    Shape const shape{2,sequence,2,4};
    int const chunks=shape.chunks();
    std::vector<int> visits(size_t(chunks),0);
    ++visits[0];
    for (int ct=0; ct<chunks; ++ct) {
      int const valid=Plan::valid(ct,sequence);
      for (int r=0; r<Chunk; ++r) {
        bad += (r<valid) != (ct*Chunk+r<sequence); ++rows;
      }
      int const next=plant==7 ? ct+1 : Plan::next(ct,chunks);
      bad += (next>=0) != (ct+1<chunks);
      if (next>=0) {
        if (next>=chunks) { ++bad; continue; }
        ++visits.at(size_t(next));
        int const picked=plant==8 ? ct : next;
        auto const group=shape.group(bh/4,bh%4,picked);
        bad += group != int64_t(bh)*chunks+ct+1;
        bad += Plan::inverse_base(group) != (int64_t(bh)*chunks+ct+1)*16384;
      }
      ++chunks_seen;
    }
    for (int n:visits) bad += n!=1;
    ++shapes;
  }
  if (!plant && (shapes!=32768 || chunks_seen!=1064960 || rows!=68157440))
    throw std::runtime_error("prefetch coverage denominator changed");
  return bad;
}
int main() {
  if (coordinates()) throw std::runtime_error("inverse prefetch tail/group/pitch mismatch");
  for (int chunks=1; chunks<=33; ++chunks) for (bool final:{false,true})
    if (lifetime(chunks,final)) throw std::runtime_error("unordered residual-prefetch shared access");
  for (int plant=1; plant<=8; ++plant) {
    auto bad=plant<=6 ? lifetime(3,true,plant) : coordinates(plant);
    if (!bad) throw std::runtime_error("prefetch negative escaped");
    std::printf("[residual prefetch negative] plant=%d bad=%llu EXPECTED-RED/PASS\n",
                plant,(unsigned long long)bad);
  }
  std::printf("[residual prefetch lifetime] shapes=32768 chunks=1064960 tail_rows=68157440 "
              "partial_order_schedules=66 shared=%zu threads=%d PASS device=NOT_RUN\n",
              sizeof(residual_prefetch::Storage),Plan::Threads);
}
