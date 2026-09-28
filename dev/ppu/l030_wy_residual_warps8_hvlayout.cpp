#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_warps8_hvlayout.cuh"

using namespace gdn_qsa::wy;
using Writer = residual_warps8_hvlayout::PublishedValue;
using Consumer = residual_warps8_hvlayout::OutputValue;
using State = residual_warps8_hvlayout::StateTile;
using Input = residual::Value;
using Output = aiu::Tile<Chunk, OutputTile::Panel>;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
static_assert(std::is_same_v<residual_warps8_hvlayout::Snapshot, residual_warps8_hlayout::Snapshot>);
static_assert(std::is_same_v<residual_warps8_hvlayout::Storage, residual_warps8_hlayout::Storage>);

enum Plant { None, OldWriter, WrongSwizzle, OldPublisher, WrongPitch,
             OldOutputAddress, WrongOutputCoordinates, MissingWarp,
             MissingSlice, MissingOutputK, MissingGroup, MissingTail,
             EarlyInputOverwrite, TransposedFinalOutput };
struct Counts { uint64_t input=0, written=0, vectors=0, reads=0, final=0; };
unsigned tag(unsigned group, unsigned t, unsigned v, unsigned valid) {
  return t < valid ? 1+group*8192+t*128+v : 0;
}
void put(std::vector<uint32_t>& words, unsigned at, unsigned value) {
  unsigned const shift=(at%2)*16;
  words.at(at/2)=(words.at(at/2)&~(65535u<<shift))|(value<<shift);
}
unsigned get(std::vector<uint32_t> const& words, unsigned at) {
  return (words.at(at/2)>>(16*(at%2)))&65535;
}

uint64_t suite(Counts& count, Plant plant=None) {
  uint64_t bad=0;
  for (unsigned valid=1;valid<=64-unsigned(plant==MissingTail);++valid) {
    std::vector<unsigned> global(2*Chunk*Dim,65535),owners(global.size(),0);
    // Two groups represent two adjacent heads for the public-output oracle.
    std::vector<unsigned> final(Chunk*Dim*2,65535),final_owners(final.size(),0);
    for (unsigned group=0;group<2-unsigned(plant==MissingGroup);++group) {
      for (unsigned slice=0;slice<4-unsigned(plant==MissingSlice);++slice) {
        std::vector<uint32_t> shared(Chunk*ValueTile/2,0);
        for (unsigned t=0;t<Chunk;++t) for (unsigned v=0;v<ValueTile;++v)
          put(shared,Input::offset(t,v),t<valid ? 32768+tag(group,t,slice*ValueTile+v,valid):0);
        if (plant==EarlyInputOverwrite)
          for (unsigned t=0;t<Chunk;++t) for (unsigned v=0;v<ValueTile;++v)
            put(shared,Writer::offset(t,v),tag(group,t,slice*ValueTile+v,valid));
        // Existing input-V reads must complete in their old layout before reuse.
        for (unsigned warp=0;warp<8;++warp) for (unsigned lane=0;lane<32;++lane)
          for (unsigned slot=0;slot<8;++slot) {
            unsigned const rc=Native::CLayout{}(lane,slot);
            unsigned const t=State::value_row(warp,0)+rc%16;
            unsigned const v=State::column(warp)+rc/16;
            bad+=get(shared,Input::offset(t,v))!=(t<valid ? 32768+tag(group,t,slice*ValueTile+v,valid):0);
            ++count.input;
          }
        std::vector<unsigned> shared_owners(Chunk*ValueTile,0);
        for (unsigned warp=0;warp<8-unsigned(plant==MissingWarp);++warp)
          for (unsigned lane=0;lane<32;++lane) for (unsigned slot=0;slot<8;++slot) {
            unsigned const rc=Native::CLayout{}(lane,slot);
            unsigned const t=State::value_row(warp,0)+rc%16;
            unsigned const v=State::column(warp)+rc/16;
            unsigned at=Writer::producer_offset(Writer::producer_base(warp,lane),0,slot);
            if (plant==OldWriter) at=Input::offset(t,v);
            if (plant==WrongSwizzle) at^=8;
            bad+=at!=Writer::offset(t,v);
            put(shared,at,tag(group,t,slice*ValueTile+v,valid));
            ++shared_owners.at(at); ++count.written;
          }
        for (auto n:shared_owners) bad+=n!=1;
        using Plan=Writer::Publication<State::Threads>;
        for (unsigned tid=0;tid<State::Threads;++tid)
          for (unsigned it=0;it<Plan::Iterations;++it) {
            unsigned const i=Plan::vector(tid,it), v=Plan::row(i), t=Plan::col(i);
            unsigned const base=Writer::offset(t,v);
            auto at=Writer::workspace_offset(group,t,slice*ValueTile+v);
            bad+=base%8!=0 || at%8!=0;
            for (unsigned x=0;x<8;++x) {
              bad+=Writer::offset(t+x,v)!=base+x;
              auto address=at+x;
              if (plant==OldPublisher) address=tile_offset(group)+(t+x)*Dim+slice*ValueTile+v;
              if (plant==WrongPitch) address=tile_offset(group)+(slice*ValueTile+v)*32+t+x;
              global.at(address)=get(shared,base+x); ++owners.at(address);
            }
            ++count.vectors;
          }
      }
      for (unsigned panel=0;panel<Dim;panel+=OutputTile::Panel) {
        using Tile=Consumer::Physical;
        auto const desc=Tile::descriptor(Chunk,OutputTile::Panel);
        bad+=desc.dim_h!=OutputTile::Panel || desc.dim_w!=Chunk || Tile::Cubes!=1;
        std::vector<uint32_t> shared(Chunk*OutputTile::Panel/2,0);
        for (unsigned v=0;v<OutputTile::Panel;++v) for (unsigned t=0;t<Chunk;++t) {
          auto at=Writer::workspace_offset(group,0,panel)+int64_t(v)*desc.dim_w+t;
          if (plant==OldOutputAddress) at=tile_offset(group)+int64_t(t)*Dim+panel+v;
          put(shared,Tile::offset(v,t),global.at(at));
        }
        for (unsigned warp=0;warp<8;++warp)
          for (unsigned t=0;t<Chunk-16*unsigned(plant==MissingOutputK);t+=16)
            for (unsigned f=0;f<OutputTile::Fragments;++f)
              for (unsigned lane=0;lane<32;++lane) {
                threadIdx.x=lane;
                unsigned const v=OutputTile::column(warp,f);
                uint32_t fragment[4];
                unsigned col=t,row=v;
                if (plant==WrongOutputCoordinates) { col=v;row=t; }
                cute::ppu_tsm_ld_swzl_sim<BF16,OutputTile::Panel,Chunk,true>(
                    fragment,shared.data(),col,row,0);
                for (unsigned slot=0;slot<8;++slot) {
                  unsigned const rc=Native::BLayout{}(lane,slot);
                  bad+=((fragment[slot/2]>>(16*(slot%2)))&65535)!=
                        tag(group,t+rc/16,panel+v+rc%16,valid);
                  ++count.reads;
                }
              }
        // Final output is a different semantic role from the private operand:
        // native C writer + old row-oriented exchange/vector publisher.
        for (unsigned warp=0;warp<8;++warp) for (unsigned f=0;f<OutputTile::Fragments;++f)
          for (unsigned lane=0;lane<32;++lane) for (unsigned slot=0;slot<8;++slot) {
            unsigned const rc=Native::CLayout{}(lane,slot);
            unsigned const t=OutputTile::row(warp)+rc%16, v=OutputTile::column(warp,f)+rc/16;
            unsigned at=Output::offset(t,v);
            if (plant==TransposedFinalOutput) at=Output::offset(v,t);
            put(shared,at,tag(group,t,panel+v,valid));
          }
        using OutPlan=StateVectorPlan<Chunk,OutputTile::Panel,2,OutputTile::Threads>;
        for (unsigned tid=0;tid<OutputTile::Threads;++tid) for (unsigned it=0;it<OutPlan::Iterations;++it) {
          unsigned const i=OutPlan::vector(tid,it), t=OutPlan::row(i), v=OutPlan::col(i);
          if (t>=valid) continue;
          unsigned const base=Output::offset(t,v);
          for (unsigned x=0;x<8;++x) {
            bad+=Output::offset(t,v+x)!=base+x;
            unsigned const at=(t*2+group)*Dim+panel+v+x;
            final.at(at)=get(shared,base+x); ++final_owners.at(at); ++count.final;
          }
        }
      }
    }
    for (unsigned group=0;group<2;++group) for (unsigned t=0;t<Chunk;++t) for (unsigned v=0;v<Dim;++v) {
      unsigned const at=group*8192+v*64+t;  // independent private ABI coordinates
      bad+=global.at(at)!=tag(group,t,v,valid) || owners.at(at)!=1;
      unsigned const out=(t*2+group)*128+v;
      bad+=final.at(out)!=(t<valid ? tag(group,t,v,valid):65535);
      bad+=final_owners.at(out)!=unsigned(t<valid);
    }
  }
  // Fixed denominators reject correct-but-incomplete work, including zero tails.
  bad+=count.input!=1048576 || count.written!=1048576 || count.vectors!=131072 ||
       count.reads!=4194304 || count.final!=532480;
  bad+=Writer::workspace_offset(int64_t(1)<<28,63,127)!=((int64_t(1)<<28)+1)*8192-1;
  return bad;
}
int main() {
  Counts count;
  if (suite(count)) throw std::runtime_error("paired Vnew lifecycle/publication/output seam mismatch");
  for (Plant plant:{OldWriter,WrongSwizzle,OldPublisher,WrongPitch,OldOutputAddress,
                    WrongOutputCoordinates,MissingWarp,MissingSlice,MissingOutputK,
                    MissingGroup,MissingTail,EarlyInputOverwrite,TransposedFinalOutput}) {
    Counts ignored;
    auto const bad=suite(ignored,plant);
    if (!bad) throw std::runtime_error("Vnew layout negative escaped");
    std::printf("[Vnew negative] plant=%d bad=%llu EXPECTED-RED/PASS\n",int(plant),(unsigned long long)bad);
  }
  std::printf("[Vnew layout] tails=64 input_reads=%llu producer_values=%llu vectors16B=%llu "
      "output_reader_values=%llu public_output_values=%llu groups=2 slices=4 "
      "native-C/B+actlize-reader exact-once=PASS private-Vt=PASS public-output=UNCHANGED device=NOT_RUN\n",
      (unsigned long long)count.input,(unsigned long long)count.written,(unsigned long long)count.vectors,
      (unsigned long long)count.reads,(unsigned long long)count.final);
}
