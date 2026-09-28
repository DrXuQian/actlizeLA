#pragma once
#include "gdn_qsa/ppu/wy_residual.cuh"

namespace gdn_qsa::wy::residual_operands {
using residual::Key;
using residual::Inverse;
using residual::Snapshot;
using residual::Value;
using residual::Storage;
using residual::Plan;

// The actual production schedule is also exercised by the host order gate.
// Slots have compile-time indices; never introduce an indirect register array.
template <int Steps, class Load, class Multiply>
CUTE_HOST_DEVICE void prefetch_atoms(Load load, Multiply multiply) {
  load(cute::Int<0>{}, cute::Int<0>{});
  cute::for_each(cute::make_seq<Steps>{}, [&](auto atom) {
    constexpr int i = decltype(atom)::value;
    if constexpr (i + 1 < Steps) load(cute::Int<i + 1>{}, cute::Int<(i + 1) % 2>{});
    multiply(atom, cute::Int<i % 2>{});
  });
}
}  // namespace gdn_qsa::wy::residual_operands
