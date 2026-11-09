#pragma once
#include <cstdint>

namespace gdn_qsa::wy::parallel_wu {
// Experimental full-C64 lifecycle, in BF16 elements:
//   solve: P in [0,4096) of each private 16384-element group
//   WU: read all P into shared, retire readers, then overwrite the SAME group
//       with W[64,128] followed by U[64,128]
//   state: consume W/U; the separate H-snapshot allocation is never aliased.
// Do not use tile_offset(group) for W: it would overwrite another group's P.
struct WorkspacePlan {
  static constexpr unsigned Chunk = 64, Dim = 128;
  static constexpr unsigned InverseElements = Chunk * Chunk;
  static constexpr unsigned PlaneElements = Chunk * Dim;
  static constexpr unsigned GroupElements = Dim * Dim;
  static constexpr std::int64_t w(std::int64_t group) { return group * GroupElements; }
  static constexpr std::int64_t u(std::int64_t group) { return w(group) + PlaneElements; }
};
static_assert(2 * WorkspacePlan::PlaneElements == WorkspacePlan::GroupElements);
static_assert(WorkspacePlan::InverseElements <= WorkspacePlan::PlaneElements);
}  // namespace gdn_qsa::wy::parallel_wu
