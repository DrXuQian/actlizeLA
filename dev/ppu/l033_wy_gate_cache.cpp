#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_gate_coefficients.cuh"

using namespace gdn_qsa::wy;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
using Tile = residual_warps8_hvlayout::StateTile;
enum class Plant { None, WrongRelative, MissingWriter, MissingTail };
bool same(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

uint64_t suite(Plant plant) {
  uint64_t bad=0, readers=0, writers=0, tails=0;
  for (int valid=1; valid<=64-int(plant==Plant::MissingTail); ++valid) {
    ++tails;
    for (float step : {0.0f, -.03125f, -4.0f}) {
      float prefix[64];
      for (int r=0; r<64; ++r) prefix[r]=step*float((r<valid ? r : valid-1)+1);
      gate_cache::Coefficients cache{};
      unsigned owners[64]={};
      for (unsigned tid=0; tid<256; ++tid)
        if (tid<64-unsigned(plant==Plant::MissingWriter)) {
          cache.publish(tid,prefix[tid],prefix[valid-1]);
          if (plant==Plant::WrongRelative)
            cache.relative[tid]=::expf(prefix[valid-1])/::expf(prefix[tid]);
          ++owners[tid]; ++writers;
        }
      for (unsigned n:owners) bad+=n!=1;
      for (unsigned tid=0; tid<256; ++tid) {
        unsigned warp=tid/32, lane=tid%32;
        for (unsigned slot=0; slot<8; ++slot) {
          // Oracle obtains rows from the real native C-layout, independently
          // of StateGateRows used by the shipping coefficient consumer.
          unsigned const row=Tile::value_row(warp,0)+Native::CLayout{}(lane,slot)%16;
          unsigned const consumer=Tile::value_row(warp,0)+
              StateGateRows::row(lane,StateGateRows::half(slot));
          bad+=consumer!=row;
          bad+=!same(cache.prefix[consumer],::expf(prefix[row]));
          bad+=!same(cache.relative[consumer],::expf(prefix[valid-1]-prefix[row]));
          ++readers;
        }
        bad+=!same(cache.prefix[valid-1],::expf(prefix[valid-1]));
      }
    }
  }
  bad+=tails!=64 || writers!=64*3*64 || readers!=64*3*256*8;
  if (plant==Plant::None)
    std::printf("[gate cache host] tails=%llu writers=%llu native-C-readers=%llu shared=45568->46080 bad=%llu\n",
      (unsigned long long)tails,(unsigned long long)writers,
      (unsigned long long)readers,(unsigned long long)bad);
  return bad;
}
int main() {
  if (suite(Plant::None)) throw std::runtime_error("gate coefficient/raw ownership failed");
  for (auto p : {Plant::WrongRelative,Plant::MissingWriter,Plant::MissingTail})
    if (!suite(p)) throw std::runtime_error("gate cache negative escaped");
  std::puts("[gate cache host] PASS negatives=3 device=NOT_RUN");
}
