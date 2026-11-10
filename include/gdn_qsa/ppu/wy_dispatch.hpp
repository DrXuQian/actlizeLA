#pragma once
#include "gdn_qsa/wy_contract.hpp"

namespace gdn_qsa::wy {

// IDs 1/2 are reserved for older development-library routes.
enum class CoreRoute : int { Retained = 0, NativeBf16 = 3 };

// Host metadata only: never read gate values, query hardware or allocate.
// Only the admitted FP32-gate prefill shapes select the native conversion.
constexpr CoreRoute core_route(Shape const& shape, bool gate_fp32) {
  return gate_fp32 && shape.batch == 1 && shape.q_heads == 16 &&
                 (shape.sequence == 2048 || shape.sequence == 8192) &&
                 (shape.value_heads == 32 || shape.value_heads == 48)
      ? CoreRoute::NativeBf16 : CoreRoute::Retained;
}

constexpr unsigned native_bf16_solve_threads(Shape const& shape) {
  return shape.sequence == 2048 && shape.value_heads == 32 ? 256 : 128;
}

}  // namespace gdn_qsa::wy
