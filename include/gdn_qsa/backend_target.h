#pragma once

// Boundary of the existing warp-MMA / legacy-AIU adapters. This is not a
// universal adapter for a future SM90 algorithm. No dependency headers here.
#if defined(GDN_QSA_TARGET_CUDA_SM90) || defined(GDN_QSA_TARGET_PPU17)
#error "SM90/PPU1.7 algorithms must not enter the legacy GDN primitive adapter"
#endif
#if defined(ACOMPUTE_VERSION) && ACOMPUTE_VERSION == 10700
#error "ACOMPUTE10700 requires the independent PPU1.7 backend, not legacy GDN"
#endif
#if defined(GDN_QSA_PPU) && defined(GDN_QSA_TARGET_CUDA_SM80)
#error "conflicting CUDA SM80 and PPU AIU backend selections"
#endif
#if defined(GDN_QSA_TARGET_PPU10) && !defined(GDN_QSA_PPU)
#error "PPU1.0 target requires its legacy AIU primitive selection"
#endif

// The existing entry keeps its historical default (CUDA SM80) when no PPU
// selection is present. New families must have separate source entrypoints.
