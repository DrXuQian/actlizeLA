// Actual pair converter + native C traits + existing shared writer/reader.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_warps8_hvlayout.cuh"
// The host compiler parses unrelated INT1/INT4 converter templates in the
// aggregate actlize header. Declare their device-only names, but do not define
// or emulate them: an accidental instantiation must still fail to link.
int __dp4a(int,int,int);
unsigned __byte_perm(unsigned,unsigned,unsigned);
__half2 __hfma2(__half2,__half2,__half2);
#include "gdn_qsa/ppu/wy_paired_conversion.cuh"

namespace {
using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
using Tile = residual_warps8_hvlayout::StateTile;
using Writer = residual_warps8_hvlayout::PublishedValue;
using Consumer = residual_warps8_hvlayout::BIntermediate;
float from_bits(uint32_t u) { float f; std::memcpy(&f,&u,4); return f; }
uint32_t bits(float f) { uint32_t u; std::memcpy(&u,&f,4); return u; }
uint16_t raw(BF16 value) { return value.raw(); }
uint16_t rne(float x) {
  uint32_t u=bits(x);
  if ((u&0x7fffffff)>0x7f800000) return 0x7fff;
  if ((u&0x7fffffff)==0x7f800000) return uint16_t(u>>16);
  return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
}
void need(bool b,char const* why) { if(!b)throw std::runtime_error(why); }
float input(unsigned group,unsigned t,unsigned v,unsigned valid) {
  if(t>=valid)return 0.0f;
  unsigned u=(group*8192+t*128+v)*2654435761u;
  return from_bits(0x3e800001u+(u%0x017fff00u)+(u&0x80000000u));
}
float gain(unsigned t) { return float(t%19+1)/23.0f; }
enum Plant { None, SwappedHalf, EarlyRound, WrongRowGain, PackedAddress, OmitTail };

uint64_t ownership(Plant plant) {
  uint64_t bad=0,values=0,readers=0,contexts=0,nonadjacent=0;
  for(unsigned valid=1;valid<=64-unsigned(plant==OmitTail);++valid)
    for(unsigned group=0;group<2;++group) for(unsigned slice=0;slice<4;++slice) {
      std::array<uint16_t,2048> unscaled{},scaled{};
      std::array<unsigned,2048> owners{};
      for(unsigned warp=0;warp<8;++warp) for(unsigned lane=0;lane<32;++lane) {
        float x[8],relative[2];
        for(unsigned half=0;half<2;++half)
          relative[half]=gain(Tile::value_row(warp,0)+lane/4+half*8);
        for(unsigned slot=0;slot<8;++slot) {
          auto const rc=Native::CLayout{}(lane,slot);
          unsigned const t=Tile::value_row(warp,0)+rc%16;
          unsigned const v=slice*32+Tile::column(warp)+rc/16;
          x[slot]=input(group,t,v,valid);
        }
        auto const got=paired_conversion::convert_new_value(x,relative);
        unsigned const base=Writer::producer_base(warp,lane);
        for(unsigned slot=0;slot<8;++slot) {
          auto const rc=Native::CLayout{}(lane,slot);
          unsigned const t=Tile::value_row(warp,0)+rc%16;
          unsigned const v=Tile::column(warp)+rc/16;
          unsigned at=Writer::producer_offset(base,0,slot);
          bad+=at!=Writer::offset(t,v) || at!=Consumer::offset(t,v);
          unsigned from=slot;
          if(plant==SwappedHalf)from^=1;
          uint16_t u=raw(got.unscaled[from]),s=raw(got.scaled[from]);
          if(plant==EarlyRound)s=rne(float(BF16(x[slot]))*gain(t));
          if(plant==WrongRowGain)s=rne(x[slot]*relative[(slot/4)^1]);
          if(plant==PackedAddress)at=Writer::producer_offset(base,0,slot&~1u)+(slot%2);
          unscaled.at(at)=u;scaled.at(at)=s;++owners.at(at);values+=2;
          if(slot%2==0)nonadjacent+=Writer::producer_offset(base,0,slot+1)!=at+1;
        }
      }
      for(unsigned t=0;t<64;++t) for(unsigned v=0;v<32;++v) {
        unsigned const at=Writer::offset(t,v);
        float const x=input(group,t,slice*32+v,valid);
        bad+=unscaled[at]!=rne(x) || scaled[at]!=rne(x*gain(t)) || owners[at]!=1;
      }
      // Exercise the actual actlize ld.swzl mapping of both B-oriented planes.
      for(unsigned row=0;row<64;row+=16) for(unsigned col=0;col<32;col+=16)
        for(unsigned lane=0;lane<32;++lane) for(int plane=0;plane<2;++plane) {
          std::array<uint32_t,128> packed{};
          auto const& sm=plane?scaled:unscaled;
          unsigned const cube=Consumer::cube(row,col)*256;
          for(unsigned i=0;i<128;++i)packed[i]=sm[cube+2*i]|(uint32_t(sm[cube+2*i+1])<<16);
          threadIdx.x=lane;
          uint32_t fragment[4];
          cute::ppu_tsm_ld_swzl_sim<BF16,16,16,true>(fragment,packed.data(),0,0,0);
          for(unsigned slot=0;slot<8;++slot) {
            auto const rc=Native::BLayout{}(lane,slot);
            unsigned const t=row+rc/16,v=slice*32+col+rc%16;
            float x=input(group,t,v,valid);
            if(plane)x*=gain(t);
            bad+=((fragment[slot/2]>>(16*(slot%2)))&65535)!=rne(x);
            ++readers;
          }
        }
      ++contexts;
    }
  bad+=contexts!=512 || values!=2097152 || readers!=2097152 || nonadjacent!=524288;
  if(plant==None)std::printf("[paired conversion host] contexts=%llu words=%llu readers=%llu same-lane-nonadjacent-pairs=%llu bad=%llu\n",
      (unsigned long long)contexts,(unsigned long long)values,(unsigned long long)readers,
      (unsigned long long)nonadjacent,(unsigned long long)bad);
  return bad;
}
}

int main(int argc,char** argv) {
  try {
    bool const omit=argc==2 && std::strcmp(argv[1],"--omit-last-tail")==0;
    need(argc==1 || omit,"unknown option");
    need(ownership(omit?OmitTail:None)==0,"paired conversion ownership/rounding/denominator");
    uint64_t words=0,early_round_witnesses=0;
    constexpr std::array<unsigned,8> lows{0,1,0x7fff,0x8000,0x8001,0xfffe,0xffff,0x1234};
    for(unsigned hi=0;hi<65536;++hi) {
      float x[8],relative[2]{0.7f,0.3f};
      for(unsigned s=0;s<8;++s)x[s]=from_bits((hi<<16)|lows[s]);
      auto const got=paired_conversion::convert_new_value(x,relative);
      for(unsigned s=0;s<8;++s) {
        need(raw(got.unscaled[s])==rne(x[s]),"unscaled RNE boundary");
        need(raw(got.scaled[s])==rne(x[s]*relative[s/4]),"scaled RNE boundary");
        need(raw(got.unscaled[s])==BF16(x[s]).raw(),"old unscaled cast differs");
        need(raw(got.scaled[s])==BF16(x[s]*relative[s/4]).raw(),"old scaled cast differs");
        early_round_witnesses+=rne(float(BF16(x[s]))*relative[s/4])!=raw(got.scaled[s]);
        words+=2;
      }
    }
    need(words==1048576 && early_round_witnesses>0,"rounding denominator/negative missing");
    for(Plant plant:{SwappedHalf,EarlyRound,WrongRowGain,PackedAddress,OmitTail}) {
      uint64_t const bad=ownership(plant);
      need(bad!=0,"negative control escaped");
      std::printf("[paired conversion negative] plant=%d bad=%llu EXPECTED-RED/PASS\n",int(plant),(unsigned long long)bad);
    }
    std::printf("[paired conversion host] rounding_words=%llu early_round_witnesses=%llu all64tails EXACT/PASS device=NOT_RUN\n",
        (unsigned long long)words,(unsigned long long)early_round_witnesses);
  } catch(std::exception const& e) { std::fprintf(stderr,"%s\n",e.what());return 1; }
}
