#pragma once
#include "primitives.cuh"
#include "gdn_qsa/wy_contract.hpp"

namespace actlize_la::ppu15 {
using gdn_qsa::wy::Shape;
constexpr int Chunk = 64, Dim = 128, ValueTile = 32;
struct Inputs {
  BF16 const *q, *k, *v, *beta;
  void const* g;
  float const* initial;
  bool gate_fp32;
  Shape shape;
};
struct Workspace {
  BF16 *inverse, *snapshots, *vnew;
  float* gates;
};
// All private pitches are ELEMENTS. P, H and Vnew have disjoint allocations.
// H/Vnew are [group,V,K] / [group,V,T], public state remains [B,Hv,K,V].
CUTE_HOST_DEVICE constexpr int64_t inverse_base(int64_t group) { return group * Chunk * Chunk; }
CUTE_HOST_DEVICE constexpr int64_t snapshot_base(int64_t group) { return group * Dim * Dim; }
CUTE_HOST_DEVICE constexpr int64_t value_base(int64_t group) { return group * Dim * Chunk; }
CUTE_DEVICE float gate(Inputs const& p, int64_t at) {
  return p.gate_fp32 ? static_cast<float const*>(p.g)[at] : float(static_cast<BF16 const*>(p.g)[at]);
}
int prepare(Inputs, Workspace, hggcStream_t);
int state(Inputs, Workspace, float*, hggcStream_t);
int output(Inputs, Workspace, BF16*, hggcStream_t);
int configure_prepare();
int configure_state();
int configure_output();
}  // namespace actlize_la::ppu15
