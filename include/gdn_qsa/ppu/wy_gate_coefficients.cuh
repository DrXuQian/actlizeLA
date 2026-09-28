#pragma once

#include "gdn_qsa/ppu/wy_residual_warps8_hvlayout.cuh"
#include <cmath>
#include <cstddef>

namespace gdn_qsa::wy::gate_cache {

// Natural-log FP32 prefix input. Keep the same rounded subtraction and expf
// as the direct reader; this is reuse, not exp2/fastmath reassociation.
struct Coefficients {
  float prefix[Chunk];
  float relative[Chunk];

  CUTE_HOST_DEVICE void publish(unsigned row, float log_prefix, float log_last) {
    prefix[row] = ::expf(log_prefix);
    relative[row] = ::expf(log_last - log_prefix);
  }
};

// Matrix planes and their AIU/SWZL offsets stay exactly as in the control.
// The extra 512 bytes and the producer's last-prefix read are explicit costs.
struct Storage {
  residual_warps8_hvlayout::Storage matrices;
  Coefficients coefficients;
};
static_assert(sizeof(Coefficients) == 512);
static_assert(offsetof(Storage, matrices) == 0);
static_assert(offsetof(Storage, coefficients) == 45568);
static_assert(sizeof(Storage) == 46080);

}  // namespace gdn_qsa::wy::gate_cache
