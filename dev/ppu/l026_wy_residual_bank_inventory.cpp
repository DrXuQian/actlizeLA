// Read-only diagnostic: production maps, not a new kernel or a hardware model.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_warps8.cuh"
#include "gdn_qsa/ppu/wy_residual_blayout.cuh"
using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
using Eight = residual_warps8::StateTile;
struct Stats { unsigned calls=0, max_words=0, max_halfwords=0, split_pairs=0, pairs=0; };
std::map<std::string,Stats> stats;
void phase(Stats& st, std::array<unsigned,32> const& bytes) {
  // An EXPLICIT hypothetical 32x4B bank model. This is address enumeration;
  // neither TSM internal phases nor ACU conflict coefficients are asserted.
  std::array<std::set<unsigned>,32> words, halves;
  for(auto byte:bytes) {
    words[(byte/4)%32].insert(byte/4);
    halves[(byte/4)%32].insert(byte/2);
  }
  for(unsigned b=0;b<32;++b) {
    st.max_words=std::max(st.max_words,unsigned(words[b].size()));
    st.max_halfwords=std::max(st.max_halfwords,unsigned(halves[b].size()));
  }
}
template<class Tile,bool Trans>
void matrix(std::string const& name,unsigned row,unsigned col) {
  auto& st=stats[name]; ++st.calls;
  std::array<std::array<unsigned,32>,8> addresses;
  for(unsigned lane=0;lane<32;++lane) for(unsigned s=0;s<8;++s) {
    unsigned v=Native::ALayout{}(lane,s);
    // ALayout and BLayout are equal as native layout objects in this atom;
    // the matrix interpretation (M,K versus N,K) determines shared coords.
    if(v!=unsigned(Native::BLayout{}(lane,s))) throw std::runtime_error("native layout changed");
    addresses[s][lane]=2*Tile::offset(row+(Trans?v/16:v%16),col+(Trans?v%16:v/16));
  }
  for(auto const& a:addresses) phase(st,a);
  for(unsigned lane=0;lane<32;++lane) for(unsigned s=0;s<8;s+=2) {
    ++st.pairs;
    st.split_pairs+=addresses[s][lane]/4!=addresses[s+1][lane]/4;
    if(!Trans && addresses[s+1][lane]!=addresses[s][lane]+2)
      throw std::runtime_error("normal matrix word is not contiguous half2");
  }
}
template<class Owners,bool H,bool BOriented=false>
void scalar(std::string const& name) {
  using Tile=std::conditional_t<H,residual::Snapshot,residual::Value>;
  constexpr unsigned Rows=H?Dim:Chunk;
  constexpr unsigned Fragments=H?Owners::KFragments:Owners::ValueFragments;
  std::vector<unsigned> owners(Rows*ValueTile);
  auto& st=stats[name];
  for(unsigned w=0;w<Owners::Threads/32;++w)
    for(unsigned f=0;f<Fragments;++f) for(unsigned s=0;s<8;++s) {
      std::array<unsigned,32> addresses;
      ++st.calls;
      for(unsigned lane=0;lane<32;++lane) {
        unsigned v=Native::CLayout{}(lane,s);
        auto rc=result_coord(lane,s);
        if(v!=unsigned(rc.row+16*rc.col)) throw std::runtime_error("native C map changed");
        unsigned row=(H?Owners::k_row(w,f):Owners::value_row(w,f))+v%16;
        unsigned col=Owners::column(w)+v/16;
        unsigned at;
        if constexpr(BOriented) at=residual_blayout::BIntermediate::offset(row,col);
        else at=Tile::offset(row,col);
        ++owners.at(at); addresses[lane]=2*at;
      }
      phase(st,addresses);
    }
  for(auto n:owners) if(n!=1) throw std::runtime_error("source denominator/owner failure");
}
template<class Owners>
void inspect(std::string name) {
  for(unsigned w=0;w<Owners::Threads/32;++w) {
    unsigned col=Owners::column(w);
    for(unsigned k=0;k<Dim;k+=16) {
      matrix<residual::Snapshot,true>(name+"/H-trans",k,col);
      for(unsigned r=0;r<Owners::ValueFragments;++r)
        matrix<residual::Key,false>(name+"/K-normal",Owners::value_row(w,r),k);
    }
    for(unsigned k=0;k<Chunk;k+=16) {
      matrix<residual::Value,true>(name+"/R-trans",k,col);
      for(unsigned r=0;r<Owners::ValueFragments;++r)
        matrix<residual::Inverse,false>(name+"/P-normal",Owners::value_row(w,r),k);
      matrix<residual::Value,true>(name+"/scaled-trans",k,col);
      for(unsigned f=0;f<Owners::KFragments;++f)
        matrix<residual::Key,true>(name+"/K-trans",k,Owners::k_row(w,f));
    }
  }
  scalar<Owners,true>(name+"/H-store");
  scalar<Owners,false>(name+"/V-load");
  scalar<Owners,false>(name+"/R-store");
  scalar<Owners,false>(name+"/Vnew-store");
  scalar<Owners,false>(name+"/scaled-store");
  // Layout-address comparison only, NOT implementation of an8-warp candidate.
  scalar<Owners,false,true>(name+"/B-oriented-store-addresses");
  auto& st=stats[name+"/final-FP32-store-once"];
  for(unsigned w=0;w<Owners::Threads/32;++w)
    for(unsigned f=0;f<Owners::KFragments;++f) for(unsigned s=0;s<8;++s) {
      ++st.calls;
      std::array<unsigned,32> addresses;
      for(unsigned lane=0;lane<32;++lane) {
        unsigned v=Native::CLayout{}(lane,s);
        addresses[lane]=4*((Owners::k_row(w,f)+v%16)*ValueTile+Owners::column(w)+v/16);
      }
      phase(st,addresses);
    }
}
template<unsigned Rows>
unsigned remove_trans() {
  using Tile=aiu::Tile<Rows,ValueTile>;
  std::vector<uint32_t> words(Rows*ValueTile/2);
  for(unsigned r=0;r<Rows;++r) for(unsigned c=0;c<ValueTile;++c) {
    unsigned at=Tile::offset(r,c);
    words[at/2]|=(r*ValueTile+c+1)<<(16*(at%2));
  }
  unsigned bad=0;
  for(unsigned r=0;r<Rows;r+=16) for(unsigned c=0;c<ValueTile;c+=16)
    for(unsigned lane=0;lane<32;++lane) {
      threadIdx.x=lane; uint32_t f[4];
      // Real actlize normal-load simulator; deliberately compare it against
      // the B-fragment required by the unchanged MMA, not our own reader.
      cute::ppu_tsm_ld_swzl_sim<BF16,Rows,ValueTile,true>(f,words.data(),c,r,0);
      for(unsigned s=0;s<8;++s) {
        unsigned v=Native::BLayout{}(lane,s);
        unsigned want=(r+v/16)*ValueTile+c+v%16+1;
        unsigned got=(f[s/2]>>(16*(s%2)))&65535u;
        bad+=got!=want;
      }
    }
  return bad;
}
int main() {
  inspect<StateTile>("warps4"); inspect<Eight>("warps8");
  std::puts("operation,calls_per_cta_chunk,max_distinct_words_per_bank32x4,max_halfwords_per_bank32x4,pairs_crossing_word,total_half2_pairs");
  for(auto const& [name,s]:stats)
    std::printf("%s,%u,%u,%u,%u,%u\n",name.c_str(),s.calls,s.max_words,s.max_halfwords,s.split_pairs,s.pairs);
  unsigned wrong=0, oob=0;
  using B=residual_blayout::BIntermediate;
  for(unsigned w=0;w<8;++w) for(unsigned lane=0;lane<32;++lane) for(unsigned s=0;s<8;++s) {
    unsigned v=Native::CLayout{}(lane,s);
    unsigned at=B::producer_offset(B::producer_base(w,lane),0,s);
    unsigned expected=B::offset(Eight::value_row(w,0)+v%16,Eight::column(w)+v/16);
    wrong+=at!=expected; oob+=at>=Chunk*ValueTile;
  }
  if(wrong!=1536 || oob!=1024) throw std::runtime_error("stale producer negative denominator changed");
  std::fprintf(stderr,"stale4warp-producer-with8warp: wrong=%u/2048 oob=%u EXPECTED-RED/PASS\n",wrong,oob);
  unsigned h=remove_trans<Dim>(),r=remove_trans<Chunk>();
  if(h!=3840 || r!=1920) throw std::runtime_error("dropped-trans negative failed");
  std::fprintf(stderr,"drop-trans-on-old-bytes: H=%u/4096 R-or-scaled=%u/2048 EXPECTED-RED/PASS\n",h,r);
}

