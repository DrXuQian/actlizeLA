#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_aiu.cuh"
#include "gdn_qsa/ppu/wy_split_prepare.cuh"

using namespace gdn_qsa::wy;
using namespace gdn_qsa::wy::split_prepare;
using Native = cute::MMA_Traits<cute::PPU0010_16x16x16_F32BF16BF16F32_TN>;
uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
float mul(float a, float b) { volatile float x = a * b; return x; }

struct Counts { uint64_t rows = 0, values = 0, products = 0, inverse = 0; };

uint64_t rows(Counts& count, unsigned plant = 0) {
  uint64_t bad = 0;
  for (unsigned fixture = 0; fixture < 8; ++fixture) for (unsigned valid = 1; valid <= 64; ++valid) {
    std::array<float,64> control{}, actual{};
    for (unsigned row = 0; row < 64; ++row) {
      float const g = row < valid ? -float((row * 37 + fixture * 17) % 71 + 1) / 1000.f : 0.f;
      control[row] = actual[row] = g;
    }
    // Independent literal legacy scan, versus the helpers used in the new
    // real kernel. In-place sequential cumsum is NOT an equivalent oracle.
    for (unsigned offset = 1; offset < 32; offset *= 2) {
      auto old = control, previous = actual;
      for (unsigned row = 0; row < 64; ++row) {
        if (row % 32 >= offset) control[row] = old[row] + old[row - offset];
        actual[row] = prefix_step(previous[row], previous[row % 32 >= offset ? row - offset : row],
                                  row % 32, offset);
      }
    }
    float const carry = actual[plant == 1 ? 30 : 31];
    for (unsigned row = 32; row < 64; ++row) control[row] += control[31];
    for (unsigned row = 0; row < 64; ++row) {
      actual[row] = prefix_carry(actual[row], carry, row);
      bad += bits(actual[row]) != bits(control[row]); ++count.rows;
      float const factor = std::exp(actual[row]);
      float const beta = row < valid ? float(BF16(float((row * 29 + fixture * 13) % 97 + 1) / 128)) : 0.f;
      for (unsigned col = 0; col < 128; ++col) {
        BF16 const x(float(int((row * 13 + col * 37 + fixture * 17) % 2047) - 1023) / 128);
        BF16 const expected(mul(mul(float(x), beta), std::exp(control[row])));
        BF16 const conditioned = plant == 2 ? BF16(mul(float(x), mul(beta, factor)))
                                 : condition_key(x, beta, factor);
        bad += conditioned.raw() != expected.raw();
        bad += condition_value(x, beta).raw() != BF16(mul(float(x), beta)).raw();
        ++count.values;
      }
    }
  }
  return bad;
}

uint64_t wu(Counts& count, unsigned plant = 0) {
  using T = aiu::Tile<64,128>;
  std::vector<unsigned> owners(8192), shared(8192, ~0u);
  std::vector<std::vector<unsigned>> reduction(8192);
  for (unsigned warp = 0; warp < 8 - unsigned(plant == 3); ++warp)
    for (unsigned f = 0; f < Plan::WUFragments; ++f)
      for (unsigned lane = 0; lane < 32; ++lane) for (unsigned slot = 0; slot < 8; ++slot) {
        auto const rc = result_coord(lane, slot);
        if (Native::CLayout{}(lane, slot) != rc.row + 16 * rc.col)
          throw std::runtime_error("MMA C layout changed");
        unsigned const row = Plan::wu_row(warp, f) + rc.row, col = Plan::wu_col(warp, f) + rc.col;
        unsigned const logical = row * 128 + col, at = T::offset(row, col) ^ (plant == 4 ? 8u : 0u);
        ++owners.at(logical); shared.at(at) = logical;
        for (unsigned k = 0; k < 64; k += 16)
          reduction.at(logical).push_back(plant == 5 ? 48 - k : k);
        count.products += 2;  // W and U independent accumulators
      }
  uint64_t bad = 0;
  for (auto n : owners) bad += n != 1;
  for (auto const& order : reduction) bad += order != std::vector<unsigned>({0,16,32,48});
  std::vector<unsigned> publish(8192), condition(8192);
  for (unsigned tid = 0; tid < Plan::WUThreads; ++tid) for (unsigned it = 0; it < Plan::ValueIterations; ++it) {
    unsigned const logical = Plan::value_vector(tid, it), row = logical / 128, col = logical % 128;
    auto const at = T::offset(row, col);
    bad += at % 8 != 0;
    for (unsigned j = 0; j < 8; ++j) {
      bad += T::offset(row, col + j) != at + j;
      bad += shared.at(at + j) != logical + j;
      ++publish.at(logical + j); ++condition.at(at + j);
    }
  }
  for (auto n : publish) bad += n != 1;
  for (auto n : condition) bad += n != 1;
  return bad;
}

uint64_t scratch(Counts& count, unsigned plant = 0) {
  uint64_t bad = 0;
  // Both sides use real group/snapshot helpers. The expected storage extents
  // and all blockId mappings are checked independently, including empty tails.
  for (int length : {1,16,63,64,65,129,2048}) for (int batch : {1,2}) for (int heads : {2,32,64}) {
    Shape const shape{batch,length,1,heads};
    auto const groups = shape.groups();
    std::vector<unsigned> writes(groups * 16384);
    for (int b = 0; b < batch; ++b) for (int h = 0; h < heads; ++h)
      for (int ct = 0; ct < (length + 63) / 64; ++ct) {
        auto const group = shape.group(b,h,ct);
        int64_t const base = plant == 6 ? tile_offset(group) : Plan::inverse_base(group);
        bad += base != ((int64_t(b) * heads + h) * ((length + 63) / 64) + ct) * 16384;
        for (unsigned tid = 0; tid < Plan::SolveThreads; ++tid)
          for (unsigned it = 0; it < Plan::InverseIterations; ++it) for (unsigned j = 0; j < 8; ++j) {
            unsigned const i = (tid + it * Plan::SolveThreads) * 8 + j;
            ++writes.at(base + i); ++count.inverse;
          }
      }
    for (int64_t group = 0; group < groups; ++group) for (unsigned i = 0; i < 16384; ++i)
      bad += writes[group * 16384 + i] != unsigned(i < 4096);
  }
  return bad;
}

int main() {
  Counts count;
  if (rows(count) || wu(count) || scratch(count)) throw std::runtime_error("split prepare contract mismatch");
  if (count.rows != 32768 || count.values != 4194304 || count.products != 16384 || count.inverse != 49373184)
    throw std::runtime_error("split prepare independent denominator mismatch");
  for (unsigned plant = 1; plant <= 6; ++plant) {
    Counts ignored;
    auto bad = plant <= 2 ? rows(ignored, plant) : plant <= 5 ? wu(ignored, plant) : scratch(ignored, plant);
    if (!bad) throw std::runtime_error("split prepare negative escaped");
    std::printf("[WY split prepare negative] plant=%u bad=%llu EXPECTED-RED/PASS\n",plant,(unsigned long long)bad);
  }
  std::printf("[WY split prepare] rows=%llu conditioned=%llu WU_outputs=%llu inverse_cells=%llu "
              "native-layout+FP32-order+vector-exact-once PASS device_execution=NOT_RUN\n",
              (unsigned long long)count.rows,(unsigned long long)count.values,
              (unsigned long long)count.products,(unsigned long long)count.inverse);
}
