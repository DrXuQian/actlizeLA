#pragma once

#include "gdn_wy_state_copy.cuh"

namespace gdn_qsa::wy {

// Extend the already-proved vector map with a tail write predicate. No global
// pointer is formed for an inactive row; unlike cp.async, a store cannot zfill.
template <int Rows, int Cols, int Threads>
CUTE_DEVICE void address_publish_tail(BF16 const* shared, BF16* global,
                                      int64_t stride, int valid_rows) {
  using Plan = StateVectorPlan<Rows, Cols, 2, Threads>;
  unsigned const tid = unsigned(threadIdx.x);
  cute::for_each(cute::make_seq<Plan::Iterations>{}, [&](auto iteration) {
    unsigned const i = Plan::vector(tid, decltype(iteration)::value);
    unsigned const row = Plan::row(i), col = Plan::col(i);
    if (row < unsigned(valid_rows)) {
      uint4 const packed = *reinterpret_cast<uint4 const*>(
          shared + state_shared_offset<Rows, Cols>(row, col));
      cutlass::arch::global_store<uint4, 16>(packed, global + int64_t(row) * stride + col, true);
    }
  });
}

}  // namespace gdn_qsa::wy
