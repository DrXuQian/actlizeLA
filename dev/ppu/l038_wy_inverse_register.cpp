// Exercise the shipping register gather against independently indexed real
// actlize C/B traits, with every tail and off-diagonal block owner.
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_inverse_register.cuh"

namespace {
using Traits = cute::MMA_Traits<cute::PPU0010_16x16x8_F32TF32TF32F32_TN>;
using Fragment = std::array<std::array<float, 8>, 32>;
uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
float value(uint32_t u) { float x; std::memcpy(&x,&u,4); return x; }
void need(bool v,char const* reason) { if(!v)throw std::runtime_error(reason); }
struct Exchange {
  Fragment const& source;
  template<int Slot> float get(unsigned lane) const { return source.at(lane).at(Slot); }
};
}

int main(int argc, char** argv) {
  try {
    bool const omit = argc == 2 && std::strcmp(argv[1],"--omit-last-block") == 0;
    need(argc==1 || omit,"unknown option");
    uint64_t contexts=0, words=0, halves=0;
    std::array<uint64_t,4> detected{};
    uint32_t rng=0x9e3779b9;
    constexpr std::array<uint32_t,8> special{0,0x80000000,0x3f800000,0xbf800000,
      0x00000001,0x7f800000,0xff800000,0x7fc12345};
    for(int tail=1;tail<=64;++tail) for(int pattern=0;pattern<8;++pattern)
      for(int gap=1;gap<4;++gap) for(int warp=0;warp<4-gap;++warp) {
        if(omit && tail==64 && pattern==7 && gap==3 && warp==0)continue;
        int const br=warp+gap,bc=warp;
        Fragment fragment;
        std::array<float,256> shared{};
        std::array<int,256> owners{};
        for(int lane=0;lane<32;++lane) for(int slot=0;slot<8;++slot) {
          int const at=Traits::CLayout{}(lane,slot), row=at%16,col=at/16;
          rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;
          uint32_t u=pattern==0?special[(row+col)%8]:rng;
          // Include tail-identity/zero cases and signed non-power-of-two bits.
          if(br*16+row>=tail || bc*16+col>=tail)u=0;
          fragment[lane][slot]=shared[row*16+col]=value(u);
          ++owners[row*16+col];
        }
        for(int count:owners)need(count==1,"C ownership is not exact-once");
        std::array<int,256> readers{};
        cute::for_each(cute::make_seq<2>{},[&](auto half) {
          constexpr int Half=decltype(half)::value;
          cute::for_each(cute::make_seq<4>{},[&](auto word) {
            constexpr int Word=decltype(word)::value;
            for(unsigned lane=0;lane<32;++lane) {
              int const at=Traits::BLayout{}(lane,Word), n=at%16,k=at/16+8*Half;
              float const want=shared[k*16+n];
              float const got=gdn_qsa::wy::inverse_register::b_value<Half,Word>(lane,Exchange{fragment});
              need(bits(got)==bits(want),"FP32 C->B bits changed");
              auto const peer=gdn_qsa::wy::inverse_register::b_owner(lane,Word,Half);
              need(bits(fragment[peer.lane][peer.slot])==bits(want),"owner formula differs from traits");
              cutlass::tfloat32_t const gh(got),wh(want);
              need(gh.raw()==wh.raw(),"TF32 high conversion input differs");
              cutlass::tfloat32_t const gl(got-float(gh)),wl(want-float(wh));
              need(gl.raw()==wl.raw(),"TF32 residual conversion input differs");
              ++readers[k*16+n];++words;halves+=2;
              detected[0]+=bits(fragment[peer.lane][peer.slot^1])!=bits(want);
              // Wrong: source chooses with its own high bit before exchange.
              int const lo=gdn_qsa::wy::inverse_register::b_owner(0,Word,Half).slot;
              detected[1]+=bits(fragment[peer.lane][lo+int(bool(peer.lane&16u))])!=bits(want);
              auto const other=gdn_qsa::wy::inverse_register::b_owner(lane,Word,1-Half);
              detected[2]+=bits(fragment[other.lane][other.slot])!=bits(want);
              detected[3]+=bits(float(cutlass::bfloat16_t(got)))!=bits(want);
            }
          });
        });
        for(int count:readers)need(count==1,"B reader coverage not exact-once");
        ++contexts;
      }
    need(contexts==3072 && words==786432 && halves==1572864,"coverage denominator incomplete");
    for(int i=0;i<4;++i) {
      need(detected[i]!=0,"wrong slot/preselection/K-half/precision negative escaped");
      std::cout<<"[inverse register negative] plant="<<i<<" mismatches="<<detected[i]<<" EXPECTED-RED/PASS\n";
    }
    std::cout<<"[inverse register host] tails=64 blocks=6 patterns=8 contexts="<<contexts
             <<" words="<<words<<" high+residual="<<halves<<" RAW-BIT/PASS device=NOT_RUN\n";
  } catch(std::exception const& e) { std::cerr<<e.what()<<'\n';return 1; }
}
