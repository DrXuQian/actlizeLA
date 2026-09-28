#pragma once
// ACOMPUTE_VERSION selects a library, not a compiler architecture.
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ != 900 || !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#error "GDN fused SM90 requires a real SM90a device compilation; no legacy/empty fallback"
#endif
#endif
#if defined(GDN_SM90_PPU17)
#if !defined(GDN_SM90_SOURCE_CHECK) && !defined(__HGGC__)
#error "native PPU1.7 requires HGGC; stock CUDA builds are source-check only"
#endif
#if !defined(ACOMPUTE_VERSION) || ACOMPUTE_VERSION != 10700
#error "PPU1.7 requires CUTLASS3.6 ACOMPUTE_VERSION=10700"
#endif
#endif
