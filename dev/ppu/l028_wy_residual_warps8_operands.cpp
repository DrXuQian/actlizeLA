#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_residual_warps8_operands.cuh"

namespace subject = gdn_qsa::wy::residual_warps8_operands;
namespace control = gdn_qsa::wy::residual_warps8_blayout;
static_assert(std::is_same_v<subject::StateTile, control::StateTile>);
static_assert(std::is_same_v<subject::BIntermediate, control::BIntermediate>);
static_assert(std::is_same_v<subject::Storage, control::Storage>);
static_assert(std::is_same_v<subject::Key, control::Key>);

enum class Plant { None, WrongSlot, MissingK, ExtraK, MissingFragment,
                   WrongWord, MissingLane, MissingWarp, EarlyOverwrite };

// Independent tags identify the complete input, not just an accumulated sum
// which could hide reordered/duplicated K steps. Exercise the actual helper.
uint64_t tag(int warp, int lane, int atom, int fragment, int word, int plane) {
  return 1 + word + 4 * (fragment + 2 * (atom + 4 * (lane + 32 * (warp + 8 * plane))));
}

template <int Steps> int check(Plant plant) {
  int bad = 0, words = 0, consumers = 0;
  for (int warp = 0; warp < (plant == Plant::MissingWarp ? 7 : 8); ++warp) {
    for (int lane = 0; lane < (plant == Plant::MissingLane ? 31 : 32); ++lane) {
      uint64_t keys[2][2][4]{}, values[2][4]{};
      bool live[2]{};
      int loaded = 0, consumed = 0;
      subject::prefetch_atoms<Steps>([&](auto atom, auto slot) {
        int const i = int(atom), s = plant == Plant::EarlyOverwrite ? 0 : int(slot);
        bad += live[s] || i != loaded++;
        live[s] = true;
        for (int k = 0; k < 2; ++k)
          for (int w = 0; w < 4; ++w) keys[s][k][w] = tag(warp,lane,i,k,w,0);
        for (int w = 0; w < 4; ++w) values[s][w] = tag(warp,lane,i,0,w,1);
      }, [&](auto atom, auto slot) {
        int const i = int(atom), s = plant == Plant::WrongSlot ? 1-int(slot) : int(slot);
        bad += !live[s] || i != consumed++;
        for (int k = 0; k < (plant == Plant::MissingFragment ? 1 : 2); ++k) {
          ++consumers;
          for (int w = 0; w < 4; ++w) {
            int const kw = plant == Plant::WrongWord ? (w + 1) % 4 : w;
            bad += keys[s][k][kw] != tag(warp,lane,i,k,w,0);
            bad += values[s][w] != tag(warp,lane,i,0,w,1);
            words += 2;
          }
        }
        live[s] = false;
      });
      bad += live[0] || live[1] || loaded != 4 || consumed != 4;
    }
  }
  bad += consumers != 2048 || words != 16384;
  return bad;
}

int main() {
  if (check<4>(Plant::None)) throw std::runtime_error("production operand schedule/denominator failed");
  for (Plant plant : {Plant::WrongSlot, Plant::MissingFragment, Plant::WrongWord,
                     Plant::MissingLane, Plant::MissingWarp, Plant::EarlyOverwrite})
    if (!check<4>(plant)) throw std::runtime_error("operand/lifetime/coverage negative escaped");
  if (!check<3>(Plant::MissingK) || !check<5>(Plant::ExtraK))
    throw std::runtime_error("K-denominator negative escaped");
  std::puts("[warps8 operands host] actual production helper: 2048 MMA consumers / 16384 operand words PASS; 8 negatives EXPECTED-RED/PASS");
}
