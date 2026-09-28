#pragma once

#include "gdn_qsa/ppu/wy_residual_warps8_blayout.cuh"
#include "gdn_qsa/ppu/wy_residual_operands.cuh"

namespace gdn_qsa::wy::residual_warps8_operands {
using residual_warps8_blayout::StateTile;
using residual_warps8_blayout::Plan;
using residual_warps8_blayout::Storage;
using residual_warps8_blayout::Key;
using residual_warps8_blayout::Inverse;
using residual_warps8_blayout::Snapshot;
using residual_warps8_blayout::Value;
using residual_warps8_blayout::BIntermediate;
using residual_operands::prefetch_atoms;
// Only UPDATE uses this schedule. KH/PR code is identical to the control;
// deeper source buffers did not survive native scheduling there.
static_assert(Plan::Threads == 256 && sizeof(Storage) == 45568);
}  // namespace gdn_qsa::wy::residual_warps8_operands
