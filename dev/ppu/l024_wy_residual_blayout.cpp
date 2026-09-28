#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_blayout.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
using B = residual_blayout::BIntermediate;

struct Counts { unsigned writes=0, reads=0, bank_phases=0; };

unsigned check(Counts& counts, unsigned plant=0) {
  std::vector<uint32_t> memory(Chunk * ValueTile / 2);
  std::vector<unsigned> owners(Chunk * ValueTile);
  unsigned bad=0;
  for (unsigned w=0; w<StateTile::Threads/32; ++w)
    for (unsigned f=0; f<StateTile::ValueFragments; ++f)
      for (unsigned lane=0; lane<32; ++lane)
        for (unsigned slot=0; slot<8; ++slot) {
          unsigned const native=Native::CLayout{}(lane,slot);
          unsigned const row=StateTile::value_row(w,f)+native%16;
          unsigned const col=StateTile::column(w)+native/16;
          unsigned at=B::producer_offset(B::producer_base(w,lane),f,slot);
          bad+=at!=B::offset(row,col);
          if (plant==1) at=residual::Value::offset(row,col); // stale producer layout
          if (plant==2) at^=8; // wrong swizzle, same shape and extent
          ++owners.at(at); ++counts.writes;
          memory.at(at/2)|=(row*ValueTile+col+1)<<(16*(at%2));
        }
  for (auto n:owners) bad+=n!=1;
  for (unsigned r=0; r<Chunk; r+=16)
    for (unsigned c=0; c<ValueTile; c+=16) {
      std::array<std::array<unsigned,32>,4> banks{};
      std::vector<uint32_t> addresses(memory.size());
      for (unsigned i=0; i<addresses.size(); ++i) addresses[i]=i;
      for (unsigned lane=0; lane<32-unsigned(plant==3); ++lane) {
        threadIdx.x=lane;
        uint32_t words[4], physical_words[4];
        // Independent authority: real actlize native-load simulator, not
        // a reader implemented by applying B::offset a second time.
        unsigned cube=B::cube(r,c);
        if (plant==5) cube^=1; // consumer points to the other V microcube
        bad+=cube!=(r/16)*(ValueTile/16)+c/16;
        cute::ppu_tsm_ld_swzl_sim<BF16,16,16,true>(
            words,memory.data()+cube*128,0,0,0);
        cute::ppu_tsm_ld_swzl_sim<BF16,16,16,true>(
            physical_words,addresses.data()+cube*128,0,0,0);
        for (unsigned slot=0; slot<8; ++slot) {
          unsigned const native=Native::BLayout{}(lane,slot);
          unsigned const row=r+native/16, col=c+native%16;
          unsigned const got=(words[slot/2]>>(16*(slot%2)))&65535u;
          bad+=got!=row*ValueTile+col+1;
          ++counts.reads;
        }
        for (unsigned word=0; word<4; ++word) {
          unsigned address=physical_words[word];
          if (plant==4) address=0; // bank model must reject collision
          ++banks[word][address%32];
        }
      }
      // Explicit model: one 32-bit word/lane phase, 32 banks of four bytes.
      // This proves address distribution, NOT hardware trans-load scheduling
      // or zero observed ACU conflicts for the whole kernel.
      for (auto const& phase:banks) {
        for (auto n:phase) bad+=n!=1;
        ++counts.bank_phases;
      }
    }
  bad+=counts.writes!=2048 || counts.reads!=2048 || counts.bank_phases!=32;
  bad+=sizeof(residual_blayout::Storage)!=sizeof(residual::Storage);
  return bad;
}

int main() {
  Counts counts;
  if (check(counts)) throw std::runtime_error("paired B layout/native fragment mismatch");
  for (unsigned plant=1; plant<=5; ++plant) {
    Counts scratch;
    unsigned const bad=check(scratch,plant);
    if (!bad) throw std::runtime_error("B-layout negative escaped");
    std::printf("[residual B-layout negative] plant=%u bad=%u EXPECTED-RED/PASS\n",plant,bad);
  }
  std::printf("[residual B-layout] writers=%u readers=%u exact-once native-C/B-traits+"
              "actlize-load-simulator PASS model-bank-phases=%u each=32-distinct-words/banks "
              "shared=45568 threads=128 grid=UNCHANGED device=NOT_RUN\n",
              counts.writes,counts.reads,counts.bank_phases);
}
