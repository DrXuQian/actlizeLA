#pragma once

// Compatibility entry for the existing warp-MMA / legacy-AIU algorithms only.
// SM90 algorithms own different primitives/pipelines and must not enter here.
#include "../../include/gdn_qsa/backend_target.h"
#if defined(GDN_QSA_PPU)
#include "../backends/ppu_aiu/primitives.cuh"
#else
#include "../backends/cuda_sm80/primitives.cuh"
#endif
