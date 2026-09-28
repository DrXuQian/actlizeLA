#pragma once

#include "gdn_wy_common.cuh"
#include "gdn_qsa/ppu/wy_state_address.cuh"

namespace gdn_qsa::wy {

// First admitted by the state ablation, also used by opt-in stage-address
// experiments. Do not change stage()/publish_*() used by admitted controls.
// Compile-time iteration offsets keep
// per-thread column/cube invariants visible to the device compiler.
template <int Rows, int Cols, int Threads>
CUTE_DEVICE void state_stage(BF16* shared, BF16 const* global, int64_t stride, int valid_rows) {
  using Plan = StateVectorPlan<Rows, Cols, 2, Threads>;
  unsigned const tid = unsigned(threadIdx.x);
  cute::for_each(cute::make_seq<Plan::Iterations>{}, [&](auto iteration) {
    unsigned const i = Plan::vector(tid, decltype(iteration)::value);
    unsigned const row = Plan::row(i), col = Plan::col(i);
    bool const valid = row < unsigned(valid_rows);
    // Retain zfill and a valid fallback pointer for inactive tail rows.
    auto const* source = global + (valid ? int64_t(row) * stride + col : 0);
    gdn_arch::async_copy16(shared + state_shared_offset<Rows, Cols>(row, col), source, valid);
  });
}

template <int Rows, int Cols, int Threads>
CUTE_DEVICE void state_publish_bf16(BF16 const* shared, BF16* global, int64_t stride) {
  using Plan = StateVectorPlan<Rows, Cols, 2, Threads>;
  unsigned const tid = unsigned(threadIdx.x);
  cute::for_each(cute::make_seq<Plan::Iterations>{}, [&](auto iteration) {
    unsigned const i = Plan::vector(tid, decltype(iteration)::value);
    unsigned const row = Plan::row(i), col = Plan::col(i);
    uint4 const packed = *reinterpret_cast<uint4 const*>(shared + state_shared_offset<Rows, Cols>(row, col));
    cutlass::arch::global_store<uint4, 16>(packed, global + int64_t(row) * stride + col, true);
  });
}

template <int Rows, int Cols, int Threads>
CUTE_DEVICE void state_publish_fp32(float const* shared, float* global, int64_t stride) {
  using Plan = StateVectorPlan<Rows, Cols, 4, Threads>;
  unsigned const tid = unsigned(threadIdx.x);
  cute::for_each(cute::make_seq<Plan::Iterations>{}, [&](auto iteration) {
    unsigned const i = Plan::vector(tid, decltype(iteration)::value);
    unsigned const row = Plan::row(i), col = Plan::col(i);
    uint4 const packed = *reinterpret_cast<uint4 const*>(shared + row * Cols + col);
    cutlass::arch::global_store<uint4, 16>(packed, global + int64_t(row) * stride + col, true);
  });
}

}  // namespace gdn_qsa::wy
