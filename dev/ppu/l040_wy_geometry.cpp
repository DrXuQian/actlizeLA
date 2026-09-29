#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_geometry.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
enum Plant { None, WrongOwner, StalePitch, WrongReader, WrongPublisher,
             MissingWarp, MissingSlice, MissingExtent };
unsigned tag(unsigned group, unsigned r, unsigned c, unsigned valid, bool snapshot) {
  return snapshot || r < valid ? 1 + group * 16384 + r * 128 + c : 0;
}
void put(std::vector<uint32_t>& words, unsigned at, unsigned x) {
  auto& word=words.at(at/2); unsigned shift=(at%2)*16;
  word=(word & ~(65535u<<shift)) | (x<<shift);
}
unsigned get(std::vector<uint32_t> const& words, unsigned at) {
  return (words.at(at/2)>>(16*(at%2)))&65535;
}

template<int Columns, int Warps, int Rows>
uint64_t check_plane(Plant plant=None) {
  using G=geometry::Config<Columns,Warps>;
  using State=typename G::StateTile;
  using Plane=typename G::template BPlane<Rows,Rows==Dim>;
  using Vectors=typename Plane::template Publication<State::Threads>;
  constexpr int Fragments=Rows==Dim ? State::KFragments : State::ValueFragments;
  uint64_t bad=0,written=0,read=0,vectors=0,output=0;
  for(unsigned valid=1;valid<=64-unsigned(plant==MissingExtent);++valid) {
    std::vector<unsigned> global(2*Rows*Dim,65535),owners(global.size(),0);
    for(unsigned group=0;group<2;++group) {
      for(unsigned slice=0;slice<Dim/Columns-unsigned(plant==MissingSlice);++slice) {
        std::vector<uint32_t> shared(Rows*Columns/2,0);
        std::vector<unsigned> assigned(Rows*Columns,0);
        for(unsigned warp=0;warp<Warps-unsigned(plant==MissingWarp);++warp)
          for(unsigned lane=0;lane<32;++lane) for(unsigned f=0;f<Fragments;++f)
            for(unsigned s=0;s<8;++s) {
              unsigned rc=Native::CLayout{}(lane,s);
              unsigned row=(Rows==Dim ? State::k_row(warp,f):State::value_row(warp,f))+rc%16;
              unsigned col=State::column(warp)+rc/16;
              unsigned at=Plane::producer_offset(Plane::producer_base(warp,lane),f,s);
              if(plant==WrongOwner) at^=8;
              if(plant==StalePitch && Columns==16) at+=(row/16)*256;
              bad+=at!=Plane::offset(row,col);
              if(at>=assigned.size()) {++bad;continue;}
              ++assigned.at(at);++written;
              put(shared,at,tag(group,row,slice*Columns+col,valid,Rows==Dim));
            }
        for(auto n:assigned) bad+=n!=1;
        // Actual native B fragment map and the actlize ld.swzl simulation.
        for(unsigned warp=0;warp<Warps;++warp) for(unsigned r=0;r<Rows;r+=16)
          for(unsigned lane=0;lane<32;++lane) {
            threadIdx.x=lane;
            unsigned cube=Plane::cube(r,State::column(warp));
            if(plant==WrongReader) cube^=1;
            uint32_t frag[4];
            cute::ppu_tsm_ld_swzl_sim<BF16,16,16,true>(frag,shared.data()+cube*128,0,0,0);
            for(unsigned s=0;s<8;++s) {
              unsigned rc=Native::BLayout{}(lane,s);
              bad+=((frag[s/2]>>(16*(s%2)))&65535)!=
                   tag(group,r+rc/16,slice*Columns+State::column(warp)+rc%16,valid,Rows==Dim);
              ++read;
            }
          }
        for(unsigned tid=0;tid<State::Threads;++tid) for(unsigned it=0;it<Vectors::Iterations;++it) {
          unsigned i=Vectors::vector(tid,it),col=Vectors::row(i),row=Vectors::col(i);
          unsigned base=Plane::offset(row,col);
          auto at=Plane::workspace_offset(group,row,slice*Columns+col);
          bad+=base%8!=0 || at%8!=0;
          if(plant==WrongPublisher) at=group*Rows*Dim+row*Dim+slice*Columns+col;
          for(unsigned x=0;x<8;++x) {
            bad+=Plane::offset(row+x,col)!=base+x;
            global.at(at+x)=get(shared,base+x);++owners.at(at+x);
          }
          ++vectors;
        }
      }
      // The SAME output consumes both geometries in [value,k/time]. Its
      // actual descriptor + physical AIU map + native B reader close the seam.
      using Tile=aiu::Tile<OutputTile::Panel,Rows>;
      for(unsigned panel=0;panel<Dim;panel+=OutputTile::Panel) {
        auto desc=Tile::descriptor(Rows,OutputTile::Panel);
        std::vector<uint32_t> sm(Rows*OutputTile::Panel/2,0);
        for(unsigned col=0;col<OutputTile::Panel;++col) for(unsigned r=0;r<Rows;++r)
          put(sm,Tile::offset(col,r),global.at(group*Rows*Dim+(panel+col)*desc.dim_w+r));
        for(unsigned warp=0;warp<8;++warp) for(unsigned r=0;r<Rows;r+=16)
          for(unsigned f=0;f<OutputTile::Fragments;++f) for(unsigned lane=0;lane<32;++lane) {
            threadIdx.x=lane;
            uint32_t frag[4];
            cute::ppu_tsm_ld_swzl_sim<BF16,OutputTile::Panel,Tile::CubeWidth,true>(
              frag,sm.data()+(r/Tile::CubeWidth)*OutputTile::Panel*Tile::CubeWidth/2,
              r%Tile::CubeWidth,OutputTile::column(warp,f),0);
            for(unsigned s=0;s<8;++s) {
              unsigned rc=Native::BLayout{}(lane,s);
              bad+=((frag[s/2]>>(16*(s%2)))&65535)!=
                   tag(group,r+rc/16,panel+OutputTile::column(warp,f)+rc%16,valid,Rows==Dim);
              ++output;
            }
          }
      }
    }
    for(unsigned group=0;group<2;++group) for(unsigned r=0;r<Rows;++r) for(unsigned c=0;c<Dim;++c) {
      unsigned at=group*Rows*128+c*Rows+r; // independent private ABI
      bad+=owners.at(at)!=1 || global.at(at)!=tag(group,r,c,valid,Rows==Dim);
    }
  }
  // Full-space denominators cannot shrink together with a faulty loop.
  bad+=written!=uint64_t(64)*2*Rows*128 || vectors!=uint64_t(64)*2*Rows*128/8;
  bad+=read!=uint64_t(64)*2*(128/Columns)*Warps*(Rows/16)*256;
  bad+=output!=uint64_t(64)*2*2*8*(Rows/16)*2*256;
  bad+=Plane::workspace_offset(int64_t(1)<<28,Rows-1,127)!=((int64_t(1)<<28)+1)*Rows*128-1;
  if(plant==None) std::printf("[geometry layout] V=%d warps=%d rows=%d written=%llu native_read=%llu output_read=%llu EXACT-ONCE/PASS\n",
    Columns,Warps,Rows,(unsigned long long)written,(unsigned long long)read,(unsigned long long)output);
  return bad;
}
template<int V,int W> void check_geometry() {
  if(check_plane<V,W,128>()+check_plane<V,W,64>()) throw std::runtime_error("geometry layout mismatch");
  for(auto plant:{WrongOwner,WrongReader,WrongPublisher,MissingWarp,MissingSlice,MissingExtent})
    if(!check_plane<V,W,128>(plant) || !check_plane<V,W,64>(plant)) throw std::runtime_error("geometry negative escaped");
  if constexpr(V==16) if(!check_plane<V,W,128>(StalePitch)||!check_plane<V,W,64>(StalePitch))
    throw std::runtime_error("stale V32 pitch negative escaped");
}
int main() {
  check_geometry<32,8>(); check_geometry<32,4>(); check_geometry<16,4>();
  std::puts("[geometry] 3/3 legal geometries x 64 extents x 2 planes x 2 groups; 38 negatives red; device=NOT_RUN PASS");
}
