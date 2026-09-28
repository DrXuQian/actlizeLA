#pragma once

#include "gdn_qsa/wy_contract.hpp"
#include <cutlass/bfloat16.h>

namespace gdn_qsa::wy::split_prepare {

// All offsets are BF16 elements. Snapshot storage is dead until state starts:
// prefix -> solve writes inverse -> WU reads inverse -> state writes snapshots.
// Stream order, NOT inter-CTA flags, closes these lifetimes.
struct Plan {
  static constexpr unsigned PrefixThreads = 64;
  static constexpr unsigned SolveThreads = 128;
  static constexpr unsigned WUThreads = 256;
  static constexpr unsigned InverseElements = Chunk * Chunk;
  static constexpr unsigned SnapshotElements = Dim * Dim;
  static constexpr unsigned VectorElements = 8;  // 16-byte global transaction
  static constexpr unsigned WUFragments = 4;     // each warp owns32x32

  static constexpr int64_t inverse_base(int64_t group) { return state_offset(group); }
  static constexpr unsigned inverse(unsigned row, unsigned col) { return row * Chunk + col; }
  static constexpr unsigned wu_row(unsigned warp, unsigned fragment) {
    return (warp / 4) * 32 + (fragment / 2) * 16;
  }
  static constexpr unsigned wu_col(unsigned warp, unsigned fragment) {
    return (warp % 4) * 32 + (fragment % 2) * 16;
  }
  static constexpr unsigned value_vector(unsigned tid, unsigned iteration) {
    return (tid + iteration * WUThreads) * VectorElements;
  }
  static constexpr unsigned ValueIterations = Chunk * Dim / VectorElements / WUThreads;
  static constexpr unsigned InverseIterations = InverseElements / VectorElements / SolveThreads;
};
static_assert(Plan::InverseElements <= Plan::SnapshotElements);
static_assert(Plan::WUThreads / 32 * Plan::WUFragments * 16 * 16 == Chunk * Dim);
static_assert(Plan::ValueIterations * Plan::WUThreads * Plan::VectorElements == Chunk * Dim);

// Small arithmetic seams are shared with the CPU raw-bit gate. They do not
// reassociate beta*factor, round a row factor, or replace the two-warp scan.
CUTLASS_HOST_DEVICE float prefix_step(float value, float peer, unsigned lane, unsigned offset) {
  if (lane >= offset) value += peer;
  return value;
}
CUTLASS_HOST_DEVICE float prefix_carry(float value, float carry, unsigned row) {
  if (row >= 32) value += carry;
  return value;
}
CUTLASS_HOST_DEVICE cutlass::bfloat16_t condition_key(
    cutlass::bfloat16_t value, float beta, float factor) {
  return cutlass::bfloat16_t(float(value) * beta * factor);
}
CUTLASS_HOST_DEVICE cutlass::bfloat16_t condition_value(cutlass::bfloat16_t value, float beta) {
  return cutlass::bfloat16_t(float(value) * beta);
}

}  // namespace gdn_qsa::wy::split_prepare
