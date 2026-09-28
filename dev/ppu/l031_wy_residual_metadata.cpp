// Exercise the production address plan, not a second implementation of GDN.
#include "gdn_qsa/wy_metadata.hpp"
#include <array>
#include <bitset>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace gdn_qsa::wy;
namespace {
void require(bool value, char const* reason) {
  if (!value) throw std::runtime_error(reason);
}
enum Plant { None, Stale, Pitch, Tail, Prologue, Publication, Denominator };
struct Counts { uint64_t contexts=0, chunks=0, gates=0, beta=0, inactive=0; };
bool same(MetadataSlot a, MetadataSlot b) {
  return a.prefix==b.prefix && a.beta==b.beta &&
         a.read_prefix==b.read_prefix && a.read_beta==b.read_beta;
}

Counts coverage(int last_sequence, Plant plant) {
  Counts counts;
  for (int sequence=1; sequence<=last_sequence; ++sequence)
    for (auto heads : {std::array<int,2>{1,1}, {1,2}, {2,4}})
      for (int b=0; b<2; ++b) for (int h=0; h<heads[1]; ++h) {
        if (plant==Denominator && b==1 && h==heads[1]-1) continue;
        Shape shape{2,sequence,heads[0],heads[1]};
        int const chunks=(sequence+63)/64;
        ++counts.contexts;
        std::array<MetadataSlot,256> pending{};
        auto fetch=[&](int ct) {
          for (unsigned tid=0; tid<256; ++tid) {
            auto at=MetadataPlan::slot(shape,b,h,ct,tid);
            if (plant==Pitch && at.read_beta) ++at.beta;
            if (plant==Tail && at.read_prefix) at.read_beta=true;
            pending[tid]=at;
            counts.gates+=at.read_prefix;
            counts.beta+=at.read_beta;
            counts.inactive+=!at.read_prefix;
          }
        };
        if (plant!=Prologue) fetch(0);
        for (int ct=0; ct<chunks; ++ct) {
          ++counts.chunks;
          for (unsigned tid=0; tid<256; ++tid) {
            int64_t token=int64_t(ct)*64+tid;
            bool gate=tid<64, beta=gate && token<sequence;
            MetadataSlot want{gate ? ((int64_t(b)*heads[1]+h)*chunks+ct)*64+tid : 0,
                              beta ? (int64_t(b)*sequence+token)*heads[1]+h : 0,
                              gate,beta};
            auto published=pending[tid];
            if (plant==Publication && tid==0) published={0,0,false,false};
            require(same(published,want),"metadata publication differs from canonical index/validity");
            if (beta) require(published.beta<int64_t(2)*sequence*heads[1],"beta OOB");
          }
          bool next=MetadataPlan::has_next(shape,ct);
          require(next==(ct+1<chunks),"last-chunk lookahead guard");
          if (next) fetch(plant==Stale ? ct : ct+1);
        }
        require(!MetadataPlan::slot(shape,b,h,chunks,0).read_prefix,"future gate OOB");
      }
  uint64_t expected_chunks=0;
  for (int n=1;n<=last_sequence;++n) expected_chunks+=(n+63)/64;
  require(counts.contexts==uint64_t(last_sequence)*14,"coverage denominator lost a context");
  require(counts.chunks==expected_chunks*14,"coverage denominator lost a chunk");
  require(counts.gates==counts.chunks*64,"gate loads duplicated/omitted");
  require(counts.beta==14ull*last_sequence*(last_sequence+1)/2,"beta loads duplicated/omitted");
  require(counts.inactive==counts.chunks*192,"non-output metadata thread denominator");
  return counts;
}

// Partial-order proof for arbitrary inter-warp order, not one CPU schedule.
// Only warps0/1 publish metadata. All8 read it, then RETIRE protects reuse.
struct Graph {
  int n=0;
  std::array<std::bitset<128>,128> before{};
  int event() { require(n<128,"graph capacity"); return n++; }
  void edge(int a,int b) { before[b].set(a); }
  void close() {
    for(int k=0;k<n;++k) for(int i=0;i<n;++i)
      if(before[i][k]) before[i]|=before[k];
  }
};
void lifetime(bool remove_retire, bool early_shared) {
  Graph graph;
  std::array<std::array<int,2>,2> publish{},prefetch{};
  std::array<std::array<int,8>,2> read{},done{};
  std::array<int,2> ready{},retire{};
  for(int c=0;c<2;++c) {
    for(int w=0;w<2;++w) { prefetch[c][w]=graph.event(); publish[c][w]=graph.event(); }
    ready[c]=graph.event(); retire[c]=graph.event();
    for(int w=0;w<8;++w) { read[c][w]=graph.event(); done[c][w]=graph.event(); }
  }
  for(int c=0;c<2;++c) {
    for(int w=0;w<2;++w) {
      graph.edge(prefetch[c][w],publish[c][w]);
      graph.edge(publish[c][w],ready[c]);
      if(c && !remove_retire && !early_shared) graph.edge(retire[c-1],publish[c][w]);
    }
    for(int w=0;w<8;++w) {
      graph.edge(ready[c],read[c][w]); graph.edge(read[c][w],done[c][w]);
      graph.edge(done[c][w],retire[c]);
      if(c==0 && w<2) graph.edge(read[c][w],prefetch[c+1][w]);
    }
  }
  graph.close();
  for(int c=0;c<2;++c) for(int reader=0;reader<8;++reader) for(int writer=0;writer<2;++writer) {
    require(graph.before[read[c][reader]][publish[c][writer]],"read before publication");
    if(c) require(graph.before[publish[c][writer]][read[c-1][reader]],"overwrote live metadata");
  }
}

template<class F> void red(char const* name,F fn) {
  try { fn(); } catch(std::runtime_error const&) {
    std::cout<<"[metadata negative] "<<name<<" EXPECTED-RED/PASS\n"; return;
  }
  throw std::runtime_error(name);
}
}

int main() {
  try {
    auto n=coverage(2048,None);
    require(n.contexts==28672 && n.chunks==473088 && n.gates==30277632 &&
            n.beta==29374464 && n.inactive==90832896,"registered exact denominator");
    Shape large{2,2147483520,1,32};
    auto at=MetadataPlan::slot(large,1,31,large.chunks()-1,63);
    int64_t token=int64_t(large.chunks()-1)*64+63;
    require(at.beta==(int64_t(large.sequence)+token)*32+31 && at.beta>INT32_MAX,
            "metadata address narrowed to32 bits");
    lifetime(false,false);
    for(auto p : {Stale,Pitch,Tail,Prologue,Publication,Denominator})
      red((std::array<char const*,7>{"none","stale-chunk","wrong-pitch","tail-read",
           "no-prologue","no-publication","missing-context"})[p],[&]{coverage(65,p);});
    red("removed-retire",[]{lifetime(true,false);});
    red("early-shared-prefetch",[]{lifetime(false,true);});
    std::cout<<"[metadata] sequences=1..2048 GVA=1:1,1:2,2:4 batch=2 contexts="<<n.contexts
             <<" chunks="<<n.chunks<<" gate_rows="<<n.gates<<" beta_rows="<<n.beta
             <<" inactive_threads="<<n.inactive<<" exact-once/lifecycle/64bit=PASS device=NOT_RUN\n";
    return 0;
  } catch(std::exception const& e) {
    std::cerr<<"[metadata] FAIL: "<<e.what()<<'\n'; return 1;
  }
}
