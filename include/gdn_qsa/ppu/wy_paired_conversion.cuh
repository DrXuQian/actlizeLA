// Copyright 2026 actlize contributors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cutlass/numeric_conversion.h>

namespace gdn_qsa::wy::paired_conversion {

struct NewValue {
  cutlass::Array<cutlass::bfloat16_t,8> unscaled;
  cutlass::Array<cutlass::bfloat16_t,8> scaled;
};

// Convert in native C-slot order, before the existing shared placement.
// Each pair is lane-local but NOT physically contiguous in that layout:
// keep the scalar publishers. Scaled values must use the original FP32 x,
// never the already rounded unscaled BF16. No reassociation or fastmath.
CUTLASS_HOST_DEVICE NewValue convert_new_value(
    float const (&x)[8], float const (&relative)[2]) {
  cutlass::Array<float,8> unscaled, scaled;
  CUTLASS_PRAGMA_UNROLL
  for (int s=0; s<8; ++s) {
    unscaled[s] = x[s];
    scaled[s] = x[s] * relative[s/4];
  }
  using Convert = cutlass::NumericArrayConverter<cutlass::bfloat16_t,float,8,
      cutlass::FloatRoundStyle::round_to_nearest>;
  return {Convert{}(unscaled), Convert{}(scaled)};
}
}  // namespace gdn_qsa::wy::paired_conversion
