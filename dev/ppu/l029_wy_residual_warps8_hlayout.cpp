#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_warps8_hlayout.cuh"

using namespace gdn_qsa::wy;
using Candidate = residual_warps8_hlayout::StateTile;
using H = residual_warps8_hlayout::Snapshot;
using OutH = residual_warps8_hlayout::OutputSnapshot;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
static_assert(std::is_same_v<residual_warps8_blayout::Storage, residual_warps8_hlayout::Storage>);
static_assert(std::is_same_v<residual_warps8_blayout::Key, residual_warps8_hlayout::Key>);

enum Plant { None, StaleSharedWriter, WrongSwizzle, WrongStateCube,
             StalePublisher, StaleOutputAddress, WrongOutputCube,
             MissingWarp, MissingSlice, MissingOutputK, MissingGroup };
struct Counts { uint64_t written=0, state_reads=0, vectors=0, output_reads=0; };
unsigned tag(unsigned group, unsigned k, unsigned v) { return 1 + group*16384 + k*128 + v; }
void put(std::vector<uint32_t>& words, unsigned at, unsigned value) {
  unsigned const shift = (at % 2) * 16;
  words.at(at/2) = (words.at(at/2) & ~(65535u << shift)) | (value << shift);
}
unsigned get(std::vector<uint32_t> const& words, unsigned at) {
  return (words.at(at/2) >> ((at % 2) * 16)) & 65535u;
}

uint64_t suite(Counts& count, Plant plant=None) {
  uint64_t bad=0;
  std::vector<unsigned> workspace(2*Dim*Dim, 0), published(workspace.size(),0);
  for (unsigned group=0;group<2-unsigned(plant==MissingGroup);++group) {
    for (unsigned slice=0;slice<4-unsigned(plant==MissingSlice);++slice) {
      std::vector<uint32_t> shared(Dim*ValueTile/2,0);
      std::vector<unsigned> owners(Dim*ValueTile,0);
      for (unsigned warp=0;warp<8-unsigned(plant==MissingWarp);++warp)
        for (unsigned lane=0;lane<32;++lane)
          for (unsigned f=0;f<Candidate::KFragments;++f)
            for (unsigned s=0;s<8;++s) {
              unsigned const native=Native::CLayout{}(lane,s);
              unsigned const k=Candidate::k_row(warp,f)+native%16;
              unsigned const v=Candidate::column(warp)+native/16;
              unsigned at=H::producer_offset(H::producer_base(warp,lane),f,s);
              if (plant==StaleSharedWriter) at=residual::Snapshot::offset(k,v);
              if (plant==WrongSwizzle) at^=8;
              bad += at!=H::offset(k,v);
              ++owners.at(at); ++count.written;
              put(shared,at,tag(group,k,slice*ValueTile+v));
            }
      for (auto n:owners) bad+=n!=1;
      // Real native B map and actlize reader, not our inverse offset.
      for (unsigned warp=0;warp<8;++warp) for (unsigned k=0;k<Dim;k+=16)
        for (unsigned lane=0;lane<32;++lane) {
          threadIdx.x=lane;
          unsigned cube=H::cube(k,Candidate::column(warp));
          if (plant==WrongStateCube) cube^=1;
          uint32_t fragment[4];
          cute::ppu_tsm_ld_swzl_sim<BF16,16,16,true>(fragment,shared.data()+cube*128,0,0,0);
          for (unsigned s=0;s<8;++s) {
            unsigned const native=Native::BLayout{}(lane,s);
            unsigned const want=tag(group,k+native/16,slice*ValueTile+Candidate::column(warp)+native%16);
            unsigned const got=(fragment[s/2]>>(16*(s%2)))&65535;
            bad+=got!=want; ++count.state_reads;
          }
        }
      using Vectors=H::Publication<Candidate::Threads>;
      for (unsigned tid=0;tid<Candidate::Threads;++tid)
        for (unsigned it=0;it<Vectors::Iterations;++it) {
          unsigned const i=Vectors::vector(tid,it), v=Vectors::row(i), k=Vectors::col(i);
          unsigned const base=H::offset(k,v);
          bad+=base%8!=0;
          auto const global=H::workspace_offset(group,k,slice*ValueTile+v);
          bad+=global%8!=0;
          for (unsigned x=0;x<8;++x) {
            bad+=H::offset(k+x,v)!=base+x;
            auto at=global+x;
            if (plant==StalePublisher) at=state_offset(group)+int64_t(k+x)*Dim+slice*ValueTile+v;
            workspace.at(at)=get(shared,base+x); ++published.at(at);
          }
          ++count.vectors;
        }
    }
    // Output consumes this exact produced workspace via its actual descriptor,
    // paired AIU layout and actlize native matrix-load simulation.
    for (unsigned panel=0;panel<Dim;panel+=OutputTile::Panel) {
      using Tile=OutH::Physical;
      auto const desc=Tile::descriptor(Dim,OutputTile::Panel);
      bad+=desc.dim_w!=Dim || desc.dim_h!=OutputTile::Panel || Tile::Cubes!=2;
      std::vector<uint32_t> shared(OutputTile::Panel*Dim/2,0);
      for (unsigned v=0;v<OutputTile::Panel;++v) for (unsigned k=0;k<Dim;++k) {
        auto at=H::workspace_offset(group,0,panel)+int64_t(v)*desc.dim_w+k;
        if (plant==StaleOutputAddress) at=state_offset(group)+int64_t(k)*Dim+panel+v;
        put(shared,Tile::offset(v,k),workspace.at(at));
      }
      for (unsigned warp=0;warp<OutputTile::Threads/32;++warp)
        for (unsigned k=0;k<Dim-16*unsigned(plant==MissingOutputK);k+=16)
          for (unsigned f=0;f<OutputTile::Fragments;++f)
            for (unsigned lane=0;lane<32;++lane) {
              threadIdx.x=lane;
              unsigned cube=k/Tile::CubeWidth;
              if (plant==WrongOutputCube) cube^=1;
              uint32_t fragment[4];
              cute::ppu_tsm_ld_swzl_sim<BF16,OutputTile::Panel,Tile::CubeWidth,true>(
                  fragment,shared.data()+cube*OutputTile::Panel*Tile::CubeWidth/2,
                  k%Tile::CubeWidth,OutputTile::column(warp,f),0);
              for (unsigned s=0;s<8;++s) {
                unsigned const native=Native::BLayout{}(lane,s);
                unsigned const want=tag(group,k+native/16,panel+OutputTile::column(warp,f)+native%16);
                unsigned const got=(fragment[s/2]>>(16*(s%2)))&65535;
                bad+=got!=want; ++count.output_reads;
              }
            }
    }
  }
  for (unsigned group=0;group<2;++group) for (unsigned k=0;k<Dim;++k)
    for (unsigned v=0;v<Dim;++v) {
      unsigned const at=group*Dim*Dim+v*Dim+k;  // independent private ABI oracle
      bad+=workspace.at(at)!=tag(group,k,v) || published.at(at)!=1;
    }
  bad+=count.written!=32768 || count.state_reads!=131072 || count.vectors!=4096 || count.output_reads!=131072;
  bad+=H::workspace_offset(int64_t(1)<<28,127,127)!=((int64_t(1)<<28)+1)*16384-1;
  return bad;
}

int main() {
  Counts counts;
  if (suite(counts)) throw std::runtime_error("paired H writer/state/global/output seam mismatch");
  for (Plant plant: {StaleSharedWriter,WrongSwizzle,WrongStateCube,StalePublisher,
                     StaleOutputAddress,WrongOutputCube,MissingWarp,MissingSlice,MissingOutputK,MissingGroup}) {
    Counts c;
    auto const bad=suite(c,plant);
    if (!bad) throw std::runtime_error("H layout negative escaped");
    std::printf("[H layout negative] plant=%d bad=%llu EXPECTED-RED/PASS\n",int(plant),(unsigned long long)bad);
  }
  std::printf("[H layout] producer_values=%llu state_reader_values=%llu vectors16B=%llu "
              "output_reader_values=%llu groups=2 slices=4 native-C/B+actlize-reader "
              "paired-private-Hvk=PASS global-publication=exact-once math=UNCHANGED device=NOT_RUN\n",
              (unsigned long long)counts.written,(unsigned long long)counts.state_reads,
              (unsigned long long)counts.vectors,(unsigned long long)counts.output_reads);
}
