#pragma once
#include "gdn_qsa/wy_contract.hpp"

namespace gdn_qsa::wy {
// Metadata addresses count ELEMENTS, not bytes: prefix is FP32, beta is BF16.
// Gate rows include chunk padding; original beta never does.
struct MetadataSlot {
  int64_t prefix, beta;
  bool read_prefix, read_beta;
};
struct MetadataPlan {
  static constexpr bool has_next(Shape shape, int chunk) {
    return chunk + 1 < shape.chunks();
  }
  static constexpr MetadataSlot slot(Shape shape, int batch, int head,
                                     int chunk, unsigned thread) {
    if (thread >= Chunk || chunk < 0 || chunk >= shape.chunks())
      return {0, 0, false, false};
    int64_t const token = int64_t(chunk) * Chunk + thread;
    bool const valid = token < shape.sequence;
    return {shape.group(batch, head, chunk) * Chunk + thread,
            valid ? (int64_t(batch) * shape.sequence + token) * shape.value_heads + head : 0,
            true, valid};
  }
};
static_assert(MetadataPlan::slot({2,65,1,2},1,1,1,0).prefix == 448);
static_assert(MetadataPlan::slot({2,65,1,2},1,1,1,0).beta == 259);
static_assert(!MetadataPlan::slot({2,65,1,2},1,1,1,1).read_beta);
static_assert(!MetadataPlan::slot({2,65,1,2},1,1,2,0).read_prefix);
}  // namespace gdn_qsa::wy
