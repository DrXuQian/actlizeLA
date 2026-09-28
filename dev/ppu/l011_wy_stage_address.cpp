#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>
static struct { int x = 0; } threadIdx;
#include "gdn_qsa/ppu/wy_tiles.cuh"
#include "gdn_qsa/ppu/wy_stage_address.cuh"

using namespace gdn_qsa::wy;

// Independent physical cube anchor; do not just compare two copies of XOR.
constexpr unsigned hardware(unsigned row, unsigned col, unsigned rows) {
  unsigned const cube = (col / 16) * (rows / 16) + row / 16;
  row %= 16; col %= 16;
  unsigned const line = row / 4, vec = ((row % 4) * 2 + col / 8) ^ (line % 2);
  return cube * 256 + (line * 32 + vec * 4 + col % 8 / 2) * 2 + col % 2;
}

struct Counts { uint64_t coordinates = 0, vectors = 0, elements = 0; };

template <unsigned Rows, unsigned Cols, unsigned Threads>
uint64_t copies(Counts& count, unsigned plant = 0) {
  using Plan = StateVectorPlan<Rows, Cols, 2, Threads>;
  uint64_t bad = 0;
  for (unsigned r = 0; r < Rows; ++r) for (unsigned c = 0; c < Cols; ++c) {
    unsigned const actual = state_shared_offset<Rows, Cols>(r, c) ^ (plant == 1 ? 8u : 0u);
    bad += actual != unsigned(swizzle<Rows, Cols>(r, c)) || actual != hardware(r, c, Rows);
    ++count.coordinates;
  }
  for (int64_t stride : {int64_t(Cols), int64_t(Cols + 8), int64_t(2048), int64_t(4096), int64_t(1) << 33}) {
    for (unsigned valid = 0; valid <= Rows; ++valid) {
      std::vector<unsigned> owners(Rows * Cols), writes(Rows * Cols);
      for (unsigned tid = 0; tid < Threads; ++tid) {
        for (unsigned it = 0; it < Plan::Iterations - (plant == 3 ? 1u : 0u); ++it) {
          unsigned const i = Plan::vector(tid, it), r = Plan::row(i), c = Plan::col(i);
          unsigned const flat = (tid + it * Threads) * 8;
          bad += r != flat / Cols || c != flat % Cols;
          bool const active = r < valid;
          bool const store = plant == 4 ? r <= valid : active;
          int64_t const offset = int64_t(r) * stride * (plant == 2 ? 2 : 1) + c;
          int64_t const expected = int64_t(flat / Cols) * stride + flat % Cols;
          bad += (active ? offset : 0) != (flat / Cols < valid ? expected : 0);
          bad += (store ? offset : -1) != (flat / Cols < valid ? expected : -1);
          unsigned const shared = state_shared_offset<Rows, Cols>(r, c);
          bad += (shared * 2) % 16 != 0;
          for (unsigned x = 0; x < 8; ++x) {
            bad += shared + x != hardware(r, c + x, Rows);
            ++owners.at(r * Cols + c + x);
            if (store) ++writes.at(r * Cols + c + x);
          }
          ++count.vectors;
        }
      }
      for (unsigned i = 0; i < owners.size(); ++i)
        bad += owners[i] != 1 || writes[i] != unsigned(i / Cols < valid);
    }
  }
  return bad;
}

uint64_t prepare_elements(Counts& count, unsigned plant = 0) {
  uint64_t bad = 0;
  std::vector<unsigned> inverse(Chunk * Chunk), values(Chunk * Dim);
  for (unsigned tid = 0; tid < ParallelThreads; ++tid) {
    for (unsigned it = 0; it < PrepareElementPlan::InverseIterations; ++it) {
      unsigned const i = PrepareElementPlan::inverse(tid, it);
      unsigned const old_i = tid + it * ParallelThreads;
      unsigned const r = i / Chunk, c = i % Chunk;
      unsigned const at = state_shared_offset<Chunk, Chunk>(r, c);
      bad += i != old_i || at != hardware(old_i / Chunk, old_i % Chunk, Chunk);
      // Upper inverse is uninitialized: conversion must not read it.
      bad += (plant == 5 ? r > c : r >= c) != (old_i / Chunk >= old_i % Chunk);
      ++inverse.at(at);
      ++count.elements;
    }
    for (unsigned it = 0; it < PrepareElementPlan::ValueIterations - (plant == 6 ? 1u : 0u); ++it) {
      unsigned const r = PrepareElementPlan::row(it), c = PrepareElementPlan::column(tid);
      unsigned const old_i = tid + it * ParallelThreads;
      unsigned const at = state_shared_offset<Chunk, Dim>(r, c);
      // Same iteration, K/V element, beta row and expf(prefix[row]) input.
      bad += r != old_i / Dim || c != old_i % Dim;
      bad += at != hardware(old_i / Dim, old_i % Dim, Chunk);
      ++values.at(at);
      ++count.elements;
    }
  }
  for (unsigned n : inverse) bad += n != 1;
  for (unsigned n : values) bad += n != 1;
  return bad;
}

uint64_t selectors(unsigned plant = 0) {
  unsigned count = 0, bad = 0;
  for (unsigned prep : {0u, 1u, 8u, 256u})
    for (unsigned state : {0u, 2u, 16u, 80u, 144u, 208u})
      for (unsigned out : {0u, 4u, 32u, 544u}) {
        if (plant == 8 && prep == 256 && state == 208 && out == 544) continue;
        unsigned const mask = prep | state | out;
        auto const actual = stage_address_selection(plant == 7 ? mask & ~768u : mask);
        bad += !valid_delivery(mask) || actual.prepare != (prep == 256) || actual.output != (out == 544);
        ++count;
      }
  return bad + (count != 96);
}

int main() {
  Counts count;
  if (copies<64,128,128>(count) || copies<64,128,256>(count) ||
      copies<128,64,256>(count) || copies<64,64,256>(count) ||
      prepare_elements(count) || selectors()) return 1;
  if (count.coordinates != 28672 || count.vectors != 1492480 || count.elements != 12288)
    throw std::runtime_error("stage address denominator differs");
  for (unsigned plant = 1; plant <= 8; ++plant) {
    Counts scratch;
    auto const bad = plant <= 4 ? copies<64,128,256>(scratch, plant) :
                     plant <= 6 ? prepare_elements(scratch, plant) : selectors(plant);
    if (!bad) throw std::runtime_error("stage address negative escaped");
    std::printf("[WY stage-address negative] plant=%u bad=%llu EXPECTED-RED/PASS\n",
                plant, static_cast<unsigned long long>(bad));
  }
  std::printf("[WY stage-address] coordinates=%llu vectors=%llu prepare_elements=%llu selectors=96 independent-cube=PASS device_execution=NOT_RUN\n",
      static_cast<unsigned long long>(count.coordinates), static_cast<unsigned long long>(count.vectors),
      static_cast<unsigned long long>(count.elements));
}
