#pragma once
#include <cutlass/cutlass.h>

namespace gdn_qsa::wy::solve_static {

// The diagonal block lives in the unchanged 64-column FP32 shared matrix.
// Instantiate every register subscript: CUTE_UNROLL alone left column[k] as
// native ivreg reads and inner backedges on SDK2.1.1. These are NOT reductions:
// each update must retain the old left-to-right subtract/multiply order.
template <int Row, int K = 0>
CUTLASS_HOST_DEVICE void subtract_row(float& x, float const* lower,
                                     float const (&column)[16]) {
  if constexpr (K < Row) {
    x -= lower[Row * 64 + K] * column[K];
    subtract_row<Row, K + 1>(x, lower, column);
  }
}

template <int Row = 0>
CUTLASS_HOST_DEVICE void diagonal(float const* lower, float* inverse,
                                 unsigned lane, float (&column)[16]) {
  if constexpr (Row < 16) {
    float x = Row == int(lane % 16) ? 1.0f : 0.0f;
    subtract_row<Row>(x, lower, column);
    column[Row] = x;
    if (lane < 16) inverse[Row * 64 + lane] = x;
    diagonal<Row + 1>(lower, inverse, lane, column);
  }
}

}  // namespace gdn_qsa::wy::solve_static
