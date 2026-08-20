/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Fixed C64 / Dk128 / Dv128 / BV64 chunked-GDN forward collective.
 *
 * The CUDA branch is a deliberately scalar, independently runnable device
 * reference.  On ppu0010 the two products whose operands both come from
 * global memory (QK^T and KK^T) use actlize's proved BF16 AIU collective and
 * the m16n16k16 BF16/BF16->FP32 atom.  Matrices produced inside the CTA use
 * the same production TiledMma through a register-resident coordinate gather;
 * this avoids inventing a register-to-swizzled-shared writer that ppu0010 does
 * not provide.  NVIDIA keeps an independent scalar device reference.
 **************************************************************************************************/
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cute/arch/copy_ppu.hpp"
#include "cute/arch/copy_ppu0010_aiu.hpp"
#include "cute/arch/mma_ppu0010.hpp"
#include "cute/ppu_tensor_mix.hpp"
#include "cute/atom/copy_traits_ppu0010_aiu.hpp"
#include "cute/atom/mma_traits_ppu0010.hpp"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_resident_mma.cuh"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_types.hpp"

// actlize's CUTLASS_DEVICE intentionally follows hgcc's compilation macros.
// Keep the independently runnable CUDA reference a real device function when
// this header is parsed by NVIDIA nvcc instead.
#if defined(__CUDACC__) && !defined(__HGGCCC__)
#define QZ_PPU_GDN_DEVICE __device__ __forceinline__
#else
#define QZ_PPU_GDN_DEVICE CUTLASS_DEVICE
#endif

// NVIDIA nvcc can compile and execute the scalar device reference without
// parsing actlize's PPU collectives.  hgcc defines __HGGCCC__ in both passes;
// only that build owns the production AIU CollectiveBuilder specialization.
#if defined(__HGGCCC__)
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/ppu_mma_aiu_multistage.hpp"
#include "cutlass/gemm/collective/builders/ppu_mma_builder.inl"
#include "cutlass/layout/layout.h"
#endif

namespace cutlass::linear_attention {

namespace detail {

enum class PpuChunkedGdnGlobalDotKind : int {
  kCausalQk,
  kStrictLowerKk,
};

#if defined(__HGGCCC__)
using PpuChunkedGdnGlobalDotTile =
    cute::Shape<cute::Int<64>, cute::Int<64>, cute::Int<64>>;
// The actlize builder uses its ClusterShape slot as the explicit warp tile.
using PpuChunkedGdnGlobalDotWarp =
    cute::Shape<cute::Int<32>, cute::Int<32>, cute::Int<64>>;

using PpuChunkedGdnGlobalDotMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::PPU0010,
    cutlass::arch::OpClassTensorOp,
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    8,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,
    8,
    float,
    PpuChunkedGdnGlobalDotTile,
    PpuChunkedGdnGlobalDotWarp,
    cute::Int<2>,
    cutlass::gemm::KernelMultistage>::CollectiveOp;

static_assert(cute::size(typename PpuChunkedGdnGlobalDotMainloop::TiledMma{}) == 128,
              "C64 PPU global-dot collective must launch four warps");
static_assert(sizeof(typename PpuChunkedGdnGlobalDotMainloop::SharedStorage) == 32768,
              "C64xC64xK64 stage-2 BF16 AIU mainloop must use exactly 32 KiB");

// The inverse uses the same PPU0010 tensor pipe at its natural block sizes.
// One warp owns each 16x16 block update; all four warps cooperate on the
// final 32x32 update.  The K-mode is carried by the fragment shape (16 or 32),
// while each physical atom remains m16n16k8 TF32.
using PpuChunkedGdnInverseTiledMma16 = cute::TiledMMA<
    cute::MMA_Atom<cute::PPU0010_16x16x8_F32TF32TF32F32_TN>,
    cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>,
                 cute::Stride<cute::_1, cute::_1, cute::_1>>,
    cute::Tile<cute::_16, cute::_16, cute::_8>>;
using PpuChunkedGdnInverseTiledMma32 = cute::TiledMMA<
    cute::MMA_Atom<cute::PPU0010_16x16x8_F32TF32TF32F32_TN>,
    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>,
                 cute::Stride<cute::_2, cute::_1, cute::_1>>,
    cute::Tile<cute::_32, cute::_32, cute::_8>>;

static_assert(cute::size(PpuChunkedGdnInverseTiledMma16{}) == 32,
              "16x16 inverse update must be one warp");
static_assert(cute::size(PpuChunkedGdnInverseTiledMma32{}) == 128,
              "32x32 inverse update must be the complete CTA");
#else
// Keeps the public type surface identical for the nvcc scalar-reference build.
struct PpuChunkedGdnGlobalDotMainloop {
  struct alignas(32) SharedStorage {
    std::uint8_t bytes[32768];
  };
};
#endif

}  // namespace detail

template <class Traits_, class Arguments_>
struct PpuChunkedGdnCollectiveBf16C64D128BV64 {
  using Traits = Traits_;
  using Arguments = Arguments_;
  using Params = Arguments;
  using Element = cutlass::bfloat16_t;
  using ElementOutput = typename Arguments::ElementOutput;
  using ElementState = typename Arguments::ElementState;
  using GlobalDotMainloop = detail::PpuChunkedGdnGlobalDotMainloop;
#if defined(__HGGCCC__)
  using ResidentMma = detail::PpuChunkedGdnResidentMmaBf16C64K64<
      typename GlobalDotMainloop::TiledMma>;
  using InverseMma16 = detail::PpuChunkedGdnResidentMmaTf32<
      detail::PpuChunkedGdnInverseTiledMma16, 16, 16, 16>;
  using InverseMma32 = detail::PpuChunkedGdnResidentMmaTf32<
      detail::PpuChunkedGdnInverseTiledMma32, 32, 32, 32>;
#endif

  static constexpr int kChunk = 64;
  static constexpr int kHeadK = 128;
  static constexpr int kHeadV = 128;
  static constexpr int kValueBlock = 64;
  static constexpr int kValueBlocks = kHeadV / kValueBlock;
  static constexpr int kThreadCount = 128;
  static constexpr int kGlobalDotAlignmentBytes = 16;
  // One CTA owns one independent BV64 slice of the KxV recurrent state.  The
  // scheduler emits two work tiles for each logical V128 head; global state
  // and output addresses remain in the original V128 layout.
  static constexpr int kStateStride = kValueBlock;
  static constexpr int kStateElements = kHeadK * kStateStride;
  static constexpr int kScoreElements = kChunk * kChunk;
  static constexpr int kWElements = kChunk * kHeadK;
  static constexpr int kValueTileElements = kChunk * kValueBlock;

  // Phase offsets are a checked liveness plan, not arbitrary scratch:
  //   [0,32K)  AIU mainloop OR {strict_lower,inverse}
  //   [0, 8K)  A_bf16 after the inverse no longer needs strict_lower
  //   [8,24K)  W, after inverse is no longer live
  //   [24,32K) U for the current V tile; generated BF16 operand after U dies
  //   [32,48K) exp(G) Q H_start; BF16 V/H operand before QH is stored
  //   [48,56K) causal gated QK^T (computed before KK^T)
  //   [56,72K) FP32 v_new; BF16 H operand before v_new is stored
  static constexpr int kPhaseBytes = 72 * 1024;
  static constexpr int kOffsetStrictLower = 0;
  static constexpr int kOffsetInverse = 16 * 1024;
  static constexpr int kOffsetA = 0;
  static constexpr int kOffsetW = 8 * 1024;
  static constexpr int kOffsetU = 24 * 1024;
  static constexpr int kOffsetOState = 32 * 1024;
  static constexpr int kOffsetP = 48 * 1024;
  static constexpr int kOffsetVNew = 56 * 1024;
  static constexpr int kBf16CubeBytes = kScoreElements * int(sizeof(Element));

  static_assert(kBf16CubeBytes == 8 * 1024,
                "generated C64xK64 BF16 operand must occupy exactly 8 KiB");
  static_assert(kOffsetA + kBf16CubeBytes <= kOffsetW &&
                    kOffsetW + 2 * kBf16CubeBytes <= kOffsetU &&
                    kOffsetU + kBf16CubeBytes <= kOffsetOState &&
                    kOffsetOState + 2 * kBf16CubeBytes <= kOffsetP &&
                    kOffsetP + kBf16CubeBytes <= kOffsetVNew &&
                    kOffsetVNew + 2 * kBf16CubeBytes <= kPhaseBytes,
                "generated-operand liveness aliases overlap a live matrix");

  // The GDN is mathematically complete in this file.  Execution denominators
  // are per BV64 work tile: QK/KK and W are intentionally recomputed by the
  // two independent V-column owners.  This is the explicit cost of buying two
  // resident 4-warp CTAs without changing the proved 128-thread MMA map.
  static constexpr bool kAllStagesConnected = true;
  static constexpr bool kGlobalQkAndKkUseAiuOnPpu0010 = true;
  static constexpr bool kGeneratedOperandMmaConnected = true;
  static constexpr bool kAllDenseForwardProductsUseAiu = true;
  static constexpr bool kInverseBlockUpdatesUseAiu = true;
  static constexpr bool kAllMatrixProductsUseAiu = true;
  static constexpr int kGeneratedProductKinds = 6;
  static constexpr int kGeneratedProductInstancesPerWorkTileChunk = 6;
  static constexpr int kGeneratedMmaPerWorkTileChunk = 640;
  static constexpr int kGlobalDotMmaPerWorkTileChunk = 256;
  static constexpr int kInverseBlockProductsPerChunk = 6;
  static constexpr int kInverseTf32MmaPerWorkTileChunk = 40;
  static constexpr int kInverseCtaBarriersPerChunk = 8;
  static constexpr int kBf16MmaPerWorkTileChunk =
      kGeneratedMmaPerWorkTileChunk + kGlobalDotMmaPerWorkTileChunk;
  static constexpr int kDenseForwardMmaPerWorkTileChunk =
      kBf16MmaPerWorkTileChunk + kInverseTf32MmaPerWorkTileChunk;
  static constexpr int kBf16MmaPerLogicalHeadChunk =
      kBf16MmaPerWorkTileChunk * kValueBlocks;
  static constexpr int kTf32MmaPerLogicalHeadChunk =
      kInverseTf32MmaPerWorkTileChunk * kValueBlocks;

  // Two-stage execution stores only BF16 values that already cross an
  // explicit BF16 boundary in the fused collective.  Copying them through
  // global workspace therefore changes neither arithmetic nor rounding.
  struct alignas(16) PreparedChunk {
    Element inverse[kScoreElements];
    Element w[kWElements];
    Element causal[kScoreElements];
  };
  // Per-(chunk,BV64) seam for the four-stage forward DAG.
  //
  // h_start is stored in the exact B-operand order consumed by ResidentMma:
  // [K64 block][value][feature-within-block].  It is BF16 because the fused
  // path explicitly rounds H before both QH and WH.  v_new remains FP32: the
  // state update rounds exp2(gamma_last-gamma) * v_new, whereas O rounds the
  // unscaled v_new.  Storing only BF16 v_new would move that rounding boundary.
  struct alignas(16) PreparedValueChunk {
    Element u[kValueTileElements];
    Element h_start[kHeadK * kValueBlock];
    float v_new[kValueTileElements];
  };
  static constexpr int kPreparedChunkElements =
      2 * kScoreElements + kWElements;
  static constexpr int kPreparedChunkBytes = int(sizeof(PreparedChunk));
  static constexpr int kPreparedValueChunkBytes =
      int(sizeof(PreparedValueChunk));
  static constexpr int kPrepareBf16MmaPerChunk =
      kGlobalDotMmaPerWorkTileChunk +
      (kGeneratedMmaPerWorkTileChunk - 512);
  static constexpr int kRecurrenceBf16MmaPerValueTileChunk = 512;
  static constexpr int kTwoStageBf16MmaPerLogicalHeadChunk =
      kPrepareBf16MmaPerChunk +
      kRecurrenceBf16MmaPerValueTileChunk * kValueBlocks;
  static constexpr int kTwoStageTf32MmaPerLogicalHeadChunk =
      kInverseTf32MmaPerWorkTileChunk;

  static_assert(kPreparedChunkElements == 16384 &&
                    kPreparedChunkBytes == 32768 &&
                    kPreparedChunkBytes ==
                        PpuChunkedGdnPreparedChunkBytes<Traits>,
                "A/W/P prepared workspace must be exactly 32 KiB per chunk");
  static_assert(kPreparedValueChunkBytes == 40960,
                "U/H-start/Vnew seam must be exactly 40 KiB per BV64 chunk");
  static_assert(kPrepareBf16MmaPerChunk == 384 &&
                    kRecurrenceBf16MmaPerValueTileChunk == 512 &&
                    kTwoStageBf16MmaPerLogicalHeadChunk == 1408 &&
                    kTwoStageTf32MmaPerLogicalHeadChunk == 40,
                "two-stage execution denominator changed");

  static_assert(Traits::ChunkSize == kChunk && Traits::HeadSizeK == kHeadK &&
                    Traits::HeadSizeV == kHeadV,
                "this collective is the fixed C64/D128 implementation");
  static_assert(std::is_same_v<typename Arguments::ElementQKV, Element> &&
                    std::is_same_v<ElementOutput, Element> &&
                    std::is_same_v<ElementState, float>,
                "v1 requires BF16 Q/K/V/O and FP32 recurrent state");

  struct SharedStorageView {
    float* state;
    float* gamma;
    float* beta;
    std::uint8_t* phase;
  };

  struct alignas(32) SharedStorage {
    // Only this CTA's BV64 state columns remain resident while chunks are the
    // outer loop.  The other half of the V128 head has a disjoint CTA owner.
    float state[kStateElements];
    float gamma[kChunk];
    float beta[kChunk];
    alignas(32) std::uint8_t phase[kPhaseBytes];

    QZ_PPU_GDN_DEVICE operator SharedStorageView() {
      return SharedStorageView{state, gamma, beta, phase};
    }
  };

  // The four-stage DAG deliberately gives each kernel only the storage that
  // is live in that stage.  Keeping the original phase offsets preserves the
  // proved resident-MMA layouts while removing dead state/arenas from the
  // launch resource contract.
  static constexpr int kPreparePhaseBytes = kOffsetP + kBf16CubeBytes;
  static constexpr int kUPhaseBytes = kOffsetOState + kBf16CubeBytes;
  static constexpr int kHPhaseBytes = kOffsetVNew + kBf16CubeBytes;
  static constexpr int kOPhaseBytes = kOffsetVNew + kBf16CubeBytes;

  struct alignas(32) PrepareSharedStorage {
    float gamma[kChunk];
    float beta[kChunk];
    alignas(32) std::uint8_t phase[kPreparePhaseBytes];

    QZ_PPU_GDN_DEVICE operator SharedStorageView() {
      return SharedStorageView{nullptr, gamma, beta, phase};
    }
  };

  struct alignas(32) USharedStorage {
    float gamma[kChunk];
    float beta[kChunk];
    alignas(32) std::uint8_t phase[kUPhaseBytes];

    QZ_PPU_GDN_DEVICE operator SharedStorageView() {
      return SharedStorageView{nullptr, gamma, beta, phase};
    }
  };

  struct alignas(32) HSharedStorage {
    float state[kStateElements];
    float gamma[kChunk];
    float beta[kChunk];
    alignas(32) std::uint8_t phase[kHPhaseBytes];

    QZ_PPU_GDN_DEVICE operator SharedStorageView() {
      return SharedStorageView{state, gamma, beta, phase};
    }
  };

  struct alignas(32) OSharedStorage {
    float gamma[kChunk];
    float beta[kChunk];
    alignas(32) std::uint8_t phase[kOPhaseBytes];

    QZ_PPU_GDN_DEVICE operator SharedStorageView() {
      return SharedStorageView{nullptr, gamma, beta, phase};
    }
  };

  static constexpr int kSharedStorageBytes = int(sizeof(SharedStorage));
  static_assert(kSharedStorageBytes == 107008,
                "C64/D128/BV64 split-V shared-memory liveness ledger changed");
  static_assert(sizeof(PrepareSharedStorage) == 57856 &&
                    sizeof(USharedStorage) == 41472 &&
                    sizeof(HSharedStorage) == 98816 &&
                    sizeof(OSharedStorage) == 66048,
                "four-stage shared-memory liveness ledger changed");
  static_assert(kSharedStorageBytes <= 262144,
                "PPU0010 exposes at most 256 KiB shared storage per CTA");
  static_assert(sizeof(typename GlobalDotMainloop::SharedStorage) <= kPhaseBytes,
                "AIU global-dot mainloop must fit the phase arena");

  CUTLASS_HOST_DEVICE static constexpr PpuChunkedGdnStatus argument_status(
      Arguments const& args) {
    PpuChunkedGdnStatus const status = can_implement_ppu_chunked_gdn<Traits>(args);
    if (status != PpuChunkedGdnStatus::kSuccess) return status;
    if ((reinterpret_cast<std::uintptr_t>(args.q) % kGlobalDotAlignmentBytes) != 0 ||
        (reinterpret_cast<std::uintptr_t>(args.k) % kGlobalDotAlignmentBytes) != 0) {
      return PpuChunkedGdnStatus::kMisalignedPointer;
    }
    return PpuChunkedGdnStatus::kSuccess;
  }

 private:
  QZ_PPU_GDN_DEVICE static float* strict_lower(SharedStorageView s) {
    return reinterpret_cast<float*>(s.phase + kOffsetStrictLower);
  }
  QZ_PPU_GDN_DEVICE static float* inverse(SharedStorageView s) {
    return reinterpret_cast<float*>(s.phase + kOffsetInverse);
  }
  QZ_PPU_GDN_DEVICE static Element* inverse_bf16(SharedStorageView s) {
    return reinterpret_cast<Element*>(s.phase + kOffsetA);
  }
  QZ_PPU_GDN_DEVICE static Element* w_matrix(SharedStorageView s) {
    return reinterpret_cast<Element*>(s.phase + kOffsetW);
  }
  QZ_PPU_GDN_DEVICE static Element* u_tile(SharedStorageView s) {
    return reinterpret_cast<Element*>(s.phase + kOffsetU);
  }
  QZ_PPU_GDN_DEVICE static float* o_state(SharedStorageView s) {
    return reinterpret_cast<float*>(s.phase + kOffsetOState);
  }
  QZ_PPU_GDN_DEVICE static Element* causal_score(SharedStorageView s) {
    return reinterpret_cast<Element*>(s.phase + kOffsetP);
  }
  QZ_PPU_GDN_DEVICE static float* v_new(SharedStorageView s) {
    return reinterpret_cast<float*>(s.phase + kOffsetVNew);
  }

  QZ_PPU_GDN_DEVICE static float exp2_gate(float x) {
    return ::exp2f(x);
  }

  // actlize's BF16 conversion members are hgcc device functions.  NVIDIA nvcc
  // still needs a runnable scalar reference, so provide the identical RNE bit
  // conversion locally instead of calling those host-only members.
  QZ_PPU_GDN_DEVICE static float to_float(Element value) {
#if defined(__CUDA_ARCH__) && !defined(__HGGCCC__)
    std::uint32_t const bits =
        std::uint32_t(*reinterpret_cast<std::uint16_t const*>(&value)) << 16;
    return __uint_as_float(bits);
#else
    return float(value);
#endif
  }

  QZ_PPU_GDN_DEVICE static Element to_bf16(float value) {
#if defined(__CUDA_ARCH__) && !defined(__HGGCCC__)
    std::uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) != 0x7f800000u) {
      bits += 0x7fffu + ((bits >> 16) & 1u);
    } else if ((bits & 0x007fffffu) != 0) {
      bits = 0x7fffffffu;
    }
    Element result{};
    *reinterpret_cast<std::uint16_t*>(&result) = std::uint16_t(bits >> 16);
    return result;
#else
    return Element(value);
#endif
  }

  QZ_PPU_GDN_DEVICE static std::int64_t qk_offset(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int local_token, int feature) {
    std::int64_t const token = std::int64_t(work.token_begin) + local_token;
    return (token * params.problem.num_qk_heads + work.qk_head_idx) * kHeadK + feature;
  }

  QZ_PPU_GDN_DEVICE static std::int64_t vo_offset(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int local_token, int value) {
    std::int64_t const token = std::int64_t(work.token_begin) + local_token;
    return (token * params.problem.num_v_heads + work.v_head_idx) * kHeadV + value;
  }

  QZ_PPU_GDN_DEVICE static std::int64_t state_offset(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int feature, int value) {
    return ((std::int64_t(work.sequence_idx) * params.problem.num_v_heads +
             work.v_head_idx) *
                kHeadK +
            feature) *
               kHeadV +
           value;
  }

  QZ_PPU_GDN_DEVICE static void load_state(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      SharedStorageView shared, int thread_idx) {
    for (int i = thread_idx; i < kStateElements; i += kThreadCount) {
      int const feature = i / kStateStride;
      int const local_value = i % kStateStride;
      int const value = work.value_begin + local_value;
      shared.state[i] = params.initial_state == nullptr
                            ? 0.0f
                            : params.initial_state[state_offset(
                                  params, work, feature, value)];
    }
    __syncthreads();
  }

  QZ_PPU_GDN_DEVICE static void store_final_state(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      SharedStorageView shared, int thread_idx) {
    if (params.final_state != nullptr) {
      for (int i = thread_idx; i < kStateElements; i += kThreadCount) {
        int const feature = i / kStateStride;
        int const local_value = i % kStateStride;
        int const value = work.value_begin + local_value;
        params.final_state[state_offset(params, work, feature, value)] = shared.state[i];
      }
    }
  }

  QZ_PPU_GDN_DEVICE static int load_chunk_scalars(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_idx, SharedStorageView shared, int thread_idx) {
    int const chunk_begin = chunk_idx * kChunk;
    int const valid = work.token_count - chunk_begin < kChunk
                          ? work.token_count - chunk_begin
                          : kChunk;
    for (int i = thread_idx; i < kChunk; i += kThreadCount) {
      if (i < valid) {
        std::int64_t const token = std::int64_t(work.token_begin) + chunk_begin + i;
        std::int64_t const gh = token * params.problem.num_v_heads + work.v_head_idx;
        shared.gamma[i] = params.gamma_log2_cumsum[gh];
        shared.beta[i] = params.beta[gh];
      } else {
        shared.gamma[i] = 0.0f;
        shared.beta[i] = 0.0f;
      }
    }
    __syncthreads();
    return valid;
  }

  QZ_PPU_GDN_DEVICE static void scalar_global_dot(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, detail::PpuChunkedGdnGlobalDotKind kind,
      SharedStorageView shared, int thread_idx) {
    float* const l = strict_lower(shared);
    Element* const p = causal_score(shared);
    for (int index = thread_idx; index < kScoreElements; index += kThreadCount) {
      int const row = index / kChunk;
      int const col = index % kChunk;
      float dot = 0.0f;
      if (row < valid && col < valid) {
        Element const* lhs = kind == detail::PpuChunkedGdnGlobalDotKind::kCausalQk
                                 ? params.q
                                 : params.k;
        for (int d = 0; d < kHeadK; ++d) {
          std::int64_t const lhs_i = qk_offset(params, work, chunk_begin + row, d);
          std::int64_t const rhs_i = qk_offset(params, work, chunk_begin + col, d);
          dot += to_float(lhs[lhs_i]) * to_float(params.k[rhs_i]);
        }
      }
      if (kind == detail::PpuChunkedGdnGlobalDotKind::kCausalQk) {
        float const value = row < valid && col < valid && row >= col
                                ? dot * exp2_gate(shared.gamma[row] - shared.gamma[col])
                                : 0.0f;
        p[index] = to_bf16(value);
      } else {
        l[index] = row < valid && col < valid && row > col
                       ? shared.beta[row] * dot *
                             exp2_gate(shared.gamma[row] - shared.gamma[col])
                       : 0.0f;
      }
    }
    __syncthreads();
  }

#if defined(__HGGCCC__)
  QZ_PPU_GDN_DEVICE static void ppu0010_global_dot(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, detail::PpuChunkedGdnGlobalDotKind kind,
      SharedStorageView shared, int thread_idx) {
    using namespace cute;
    using X = Underscore;
    using Mainloop = GlobalDotMainloop;
    using Tile = detail::PpuChunkedGdnGlobalDotTile;
    using StrideA = typename Mainloop::StrideA;
    using StrideB = typename Mainloop::StrideB;

    Element const* const lhs =
        (kind == detail::PpuChunkedGdnGlobalDotKind::kCausalQk ? params.q : params.k) +
        qk_offset(params, work, chunk_begin, 0);
    Element const* const rhs = params.k + qk_offset(params, work, chunk_begin, 0);
    std::int64_t const row_pitch = std::int64_t(params.problem.num_qk_heads) * kHeadK;
    StrideA const dA{row_pitch, Int<1>{}, Int<0>{}};
    StrideB const dB{row_pitch, Int<1>{}, Int<0>{}};
    auto const problem_shape =
        make_shape(valid, valid, Int<kHeadK>{}, Int<1>{});
    typename Mainloop::Arguments mainloop_args{lhs, dA, rhs, dB};
    auto const mainloop_params = Mainloop::to_underlying_arguments(
        problem_shape, mainloop_args, nullptr);

    // make_mix_tensor_like is required: the collective constructor initializes
    // its descriptor with a null gmem pointer, and the mix iterator is what
    // carries the real per-work-tile pointer into Copy_Traits.
    Tensor mA_mkl = make_tensor(
        make_gmem_ptr(mainloop_params.ptr_A),
        make_shape(valid, Int<kHeadK>{}, Int<1>{}),
        mainloop_params.dA);
    Tensor mB_nkl = make_tensor(
        make_gmem_ptr(mainloop_params.ptr_B),
        make_shape(valid, Int<kHeadK>{}, Int<1>{}),
        mainloop_params.dB);
    Tensor mA_mk = make_mix_tensor_like(mA_mkl(_, _, 0));
    Tensor mB_nk = make_mix_tensor_like(mB_nkl(_, _, 0));
    auto const block_coord = make_coord(0, 0, _, 0);
    Tensor gA = local_tile(
        mA_mk, Tile{}, take<0, 3>(block_coord), Step<_1, X, _1>{});
    Tensor gB = local_tile(
        mB_nk, Tile{}, take<0, 3>(block_coord), Step<X, _1, _1>{});

    typename Mainloop::TiledMma tiled_mma;
    Tensor accum = partition_fragment_C(tiled_mma, make_shape(Int<64>{}, Int<64>{}));
    clear(accum);
    auto k_tile_iter = make_coord_iterator(shape<2>(gA));
    int k_tile_count = size<2>(gA);
    auto const residue = make_tuple(
        valid, valid, Int<kHeadK>{} - size<1>(gA) * size<2>(gA));
    Mainloop mainloop(mainloop_params, problem_shape);
    mainloop(
        accum, gA, gB, accum, k_tile_iter, k_tile_count, residue, thread_idx,
        reinterpret_cast<char*>(shared.phase));

    auto identity = make_identity_tensor(make_shape(Int<64>{}, Int<64>{}));
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    auto coord = thread_mma.partition_C(identity);
    float* const l = strict_lower(shared);
    Element* const p = causal_score(shared);
    for (int i = 0; i < int(size(accum)); ++i) {
      int const row = int(get<0>(coord(i)));
      int const col = int(get<1>(coord(i)));
      float const dot = accum(i);
      if (kind == detail::PpuChunkedGdnGlobalDotKind::kCausalQk) {
        float const value = row < valid && col < valid && row >= col
                                ? dot * exp2_gate(shared.gamma[row] - shared.gamma[col])
                                : 0.0f;
        p[row * kChunk + col] = to_bf16(value);
      } else {
        l[row * kChunk + col] = row < valid && col < valid && row > col
                                    ? shared.beta[row] * dot *
                                          exp2_gate(shared.gamma[row] - shared.gamma[col])
                                    : 0.0f;
      }
    }
    __syncthreads();
  }
#endif

  QZ_PPU_GDN_DEVICE static void global_dot(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, detail::PpuChunkedGdnGlobalDotKind kind,
      SharedStorageView shared, int thread_idx) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_global_dot(params, work, chunk_begin, valid, kind, shared, thread_idx);
#elif defined(__HGGC_ARCH__)
    (void)params;
    (void)work;
    (void)chunk_begin;
    (void)valid;
    (void)kind;
    (void)shared;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("PPU chunked GDN supports only ppu0010");
#else
    scalar_global_dot(params, work, chunk_begin, valid, kind, shared, thread_idx);
#endif
  }

  QZ_PPU_GDN_DEVICE static void scalar_solve_inverse(
      SharedStorageView shared, int thread_idx) {
    float* const l = strict_lower(shared);
    float* const a = inverse(shared);
    // One thread per RHS column; rows are a true dependency chain.
    for (int row = 0; row < kChunk; ++row) {
      if (thread_idx < kChunk) {
        int const col = thread_idx;
        float value = row == col ? 1.0f : 0.0f;
        for (int k = 0; k < row; ++k) {
          value -= l[row * kChunk + k] * a[k * kChunk + col];
        }
        a[row * kChunk + col] = value;
      }
      __syncthreads();
    }
    Element* const ab = inverse_bf16(shared);
    for (int i = thread_idx; i < kScoreElements; i += kThreadCount) {
      ab[i] = to_bf16(a[i]);
    }
    __syncthreads();
  }

#if defined(__HGGCCC__)
  // Invert the four diagonal 16x16 blocks directly.  These are the only true
  // triangular dependency chains.  Sixty-four threads own (block,column): a
  // complete RHS column remains on one thread, so rows of that column are a
  // register-ordered dependency rather than a CTA dependency.  One barrier
  // after all four blocks is sufficient before the block GEMMs consume them.
  QZ_PPU_GDN_DEVICE static void ppu0010_inverse_base16(
      SharedStorageView shared, int thread_idx) {
    float* const l = strict_lower(shared);
    float* const a = inverse(shared);
    for (int index = thread_idx; index < kScoreElements;
         index += kThreadCount) {
      a[index] = 0.0f;
    }
    __syncthreads();

    int const block = thread_idx / 16;
    int const column = thread_idx % 16;
    int const begin = block * 16;
    for (int row = 0; row < 16; ++row) {
      if (thread_idx < 64) {
        float value = row == column ? 1.0f : 0.0f;
        for (int k = 0; k < row; ++k) {
          value -= l[(begin + row) * kChunk + begin + k] *
                   a[(begin + k) * kChunk + begin + column];
        }
        a[(begin + row) * kChunk + begin + column] = value;
      }
    }
    __syncthreads();
  }

  // Build both 32x32 diagonal inverses concurrently.  Warp 0 owns rows/cols
  // [0,32), warp 1 owns [32,64); warps 2/3 are intentionally idle because a
  // 16x16 TF32 update is exactly one physical warp.  The dead strict-lower C
  // block is reused for D^-1*C after every source value is resident.
  QZ_PPU_GDN_DEVICE static void ppu0010_inverse_16_to_32(
      SharedStorageView shared, int thread_idx) {
    float* const l = strict_lower(shared);
    float* const a = inverse(shared);
    int const warp = thread_idx / 32;
    int const lane = thread_idx % 32;
    if (warp < 2) {
      int const begin = warp * 32;
      int const lower = begin + 16;
      auto accum = InverseMma16::make_accumulator();
      InverseMma16::clear(accum);
      InverseMma16::mma(
          accum,
          a + lower * kChunk + lower, kChunk, 1,
          l + lower * kChunk + begin, 1, kChunk,
          lane);
      InverseMma16::visit_output(
          accum, lane,
          [&](int row, int column, float value) {
            l[(lower + row) * kChunk + begin + column] = value;
          });
    }
    __syncthreads();

    if (warp < 2) {
      int const begin = warp * 32;
      int const lower = begin + 16;
      auto accum = InverseMma16::make_accumulator();
      InverseMma16::clear(accum);
      InverseMma16::mma(
          accum,
          l + lower * kChunk + begin, kChunk, 1,
          a + begin * kChunk + begin, 1, kChunk,
          lane);
      InverseMma16::visit_output(
          accum, lane,
          [&](int row, int column, float value) {
            a[(lower + row) * kChunk + begin + column] = -value;
          });
    }
    __syncthreads();
  }

  // Merge the two 32x32 diagonal inverses into the final C64 inverse.  All
  // four warps participate.  A CTA barrier separates fragment gathering from
  // overwriting the strict-lower C block because A/B values are duplicated
  // across the 2M x 2N warp topology.
  QZ_PPU_GDN_DEVICE static void ppu0010_inverse_32_to_64(
      SharedStorageView shared, int thread_idx) {
    float* const l = strict_lower(shared);
    float* const a = inverse(shared);
    constexpr int lower = 32;
    {
      auto accum = InverseMma32::make_accumulator();
      InverseMma32::clear(accum);
      InverseMma32::mma(
          accum,
          a + lower * kChunk + lower, kChunk, 1,
          l + lower * kChunk, 1, kChunk,
          thread_idx);
      __syncthreads();
      InverseMma32::visit_output(
          accum, thread_idx,
          [&](int row, int column, float value) {
            l[(lower + row) * kChunk + column] = value;
          });
    }
    __syncthreads();

    {
      auto accum = InverseMma32::make_accumulator();
      InverseMma32::clear(accum);
      InverseMma32::mma(
          accum,
          l + lower * kChunk, kChunk, 1,
          a, 1, kChunk,
          thread_idx);
      InverseMma32::visit_output(
          accum, thread_idx,
          [&](int row, int column, float value) {
            a[(lower + row) * kChunk + column] = -value;
          });
    }
    __syncthreads();
  }

  QZ_PPU_GDN_DEVICE static void ppu0010_solve_inverse(
      SharedStorageView shared, int thread_idx) {
    ppu0010_inverse_base16(shared, thread_idx);
    ppu0010_inverse_16_to_32(shared, thread_idx);
    ppu0010_inverse_32_to_64(shared, thread_idx);
    Element* const ab = inverse_bf16(shared);
    float const* const a = inverse(shared);
    for (int i = thread_idx; i < kScoreElements; i += kThreadCount) {
      ab[i] = to_bf16(a[i]);
    }
    __syncthreads();
  }
#endif

  QZ_PPU_GDN_DEVICE static void solve_inverse(
      SharedStorageView shared, int thread_idx) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_solve_inverse(shared, thread_idx);
#elif defined(__HGGC_ARCH__)
    (void)shared;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("blocked GDN inverse requires ppu0010");
#else
    scalar_solve_inverse(shared, thread_idx);
#endif
  }

  QZ_PPU_GDN_DEVICE static void compute_w(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, SharedStorageView shared, int thread_idx) {
    Element const* const a = inverse_bf16(shared);
    Element* const w = w_matrix(shared);
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    // The U tile is not live until compute_value_tile.  Use it as one C64xK64
    // BF16 materialization buffer for beta*exp2(gamma)*K, preserving the exact
    // pre-MMA rounding boundary of the scalar reference.
    Element* const scaled_k = u_tile(shared);
#pragma unroll
    for (int feature_base = 0; feature_base < kHeadK;
         feature_base += kChunk) {
      for (int index = thread_idx; index < kScoreElements;
           index += kThreadCount) {
        int const feature = index / kChunk;
        int const row = index % kChunk;
        scaled_k[index] = to_bf16(
            row < valid
                ? shared.beta[row] * exp2_gate(shared.gamma[row]) *
                      to_float(params.k[qk_offset(
                          params, work, chunk_begin + row,
                          feature_base + feature)])
                : 0.0f);
      }
      __syncthreads();

      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
      ResidentMma::mma(
          accum,
          a, kChunk, 1,
          scaled_k, kChunk, 1,
          thread_idx);
      // All operand fragments must be resident before the next block can
      // overwrite scaled_k.  This is a source-lifetime barrier, not an MMA
      // synchronization requirement.
      __syncthreads();
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int row, int feature, float value) {
            w[row * kHeadK + feature_base + feature] = to_bf16(value);
          });
      __syncthreads();
    }
#elif defined(__HGGC_ARCH__)
    (void)params;
    (void)work;
    (void)chunk_begin;
    (void)valid;
    (void)shared;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("generated GDN MMA requires ppu0010");
#else
    for (int index = thread_idx; index < kWElements; index += kThreadCount) {
      int const row = index / kHeadK;
      int const d = index % kHeadK;
      float sum = 0.0f;
      for (int j = 0; j < kChunk; ++j) {
        // Upstream forms the scaled K operand in the input dtype before the
        // A@K tensor product.  Keep that BF16 boundary explicit.
        Element const kb = to_bf16(
            j < valid
                ? shared.beta[j] * exp2_gate(shared.gamma[j]) *
                      to_float(params.k[qk_offset(
                          params, work, chunk_begin + j, d)])
                : 0.0f);
        sum += to_float(a[row * kChunk + j]) * to_float(kb);
      }
      w[index] = to_bf16(sum);
    }
    __syncthreads();
#endif
  }

#if defined(__HGGCCC__)
  QZ_PPU_GDN_DEVICE static void ppu0010_compute_value_tile(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, int value_base,
      SharedStorageView shared, int thread_idx) {
    Element const* const a = inverse_bf16(shared);
    Element const* const w = w_matrix(shared);
    Element* const u = u_tile(shared);
    float* const os = o_state(shared);
    Element const* const p = causal_score(shared);
    float* const vn = v_new(shared);
    std::int64_t const qk_row_pitch =
        std::int64_t(params.problem.num_qk_heads) * kHeadK;

    // U = A @ round_bf16(beta * V).  Before QH is produced, the first 8 KiB
    // of its FP32 destination is a dead C64xK64 BF16 operand arena.
    Element* const generated_operand =
        reinterpret_cast<Element*>(shared.phase + kOffsetOState);
    for (int index = thread_idx; index < kScoreElements;
         index += kThreadCount) {
      int const value = index / kChunk;
      int const row = index % kChunk;
      generated_operand[index] = to_bf16(
          row < valid
              ? shared.beta[row] *
                    to_float(params.v[vo_offset(
                        params, work, chunk_begin + row,
                        value_base + value)])
              : 0.0f);
    }
    __syncthreads();
    {
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
      ResidentMma::mma(
          accum,
          a, kChunk, 1,
          generated_operand, kChunk, 1,
          thread_idx, valid, kValueBlock, kChunk);
      __syncthreads();
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int row, int value, float product) {
            u[row * kValueBlock + value] = to_bf16(product);
          });
    }
    __syncthreads();

    // QH: retain one FP32 accumulator while two K64 state cubes are consumed.
    // The H materialization shares storage with os only after a barrier proves
    // every thread has gathered the final cube into registers.
    {
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
#pragma unroll
      for (int feature_base = 0; feature_base < kHeadK;
           feature_base += kChunk) {
        for (int index = thread_idx; index < kScoreElements;
             index += kThreadCount) {
          int const value = index / kChunk;
          int const feature = index % kChunk;
          generated_operand[index] = to_bf16(
              shared.state[(feature_base + feature) * kStateStride + value]);
        }
        __syncthreads();
        Element const* const q =
            params.q + qk_offset(
                           params, work, chunk_begin, feature_base);
        ResidentMma::mma(
            accum,
            q, qk_row_pitch, 1,
            generated_operand, kChunk, 1,
            thread_idx, valid, kValueBlock, kChunk);
        __syncthreads();
      }
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int row, int value, float product) {
            os[row * kValueBlock + value] =
                row < valid ? exp2_gate(shared.gamma[row]) * product : 0.0f;
          });
    }
    __syncthreads();

    // WH uses the same rounded H boundary but a separate accumulator so QH
    // and WH do not double the live FP32 register footprint.  The destination
    // v_new region is the temporary H arena until the final gather completes.
    Element* const h_for_wh =
        reinterpret_cast<Element*>(shared.phase + kOffsetVNew);
    {
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
#pragma unroll
      for (int feature_base = 0; feature_base < kHeadK;
           feature_base += kChunk) {
        for (int index = thread_idx; index < kScoreElements;
             index += kThreadCount) {
          int const value = index / kChunk;
          int const feature = index % kChunk;
          h_for_wh[index] = to_bf16(
              shared.state[(feature_base + feature) * kStateStride + value]);
        }
        __syncthreads();
        ResidentMma::mma(
            accum,
            w + feature_base, kHeadK, 1,
            h_for_wh, kChunk, 1,
            thread_idx, valid, kValueBlock, kChunk);
        __syncthreads();
      }
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int row, int value, float product) {
            vn[row * kValueBlock + value] =
                row < valid
                    ? to_float(u[row * kValueBlock + value]) - product
                    : 0.0f;
          });
    }
    __syncthreads();

    // P @ round_bf16(Vnew).  U is dead after Vnew has been formed, so its
    // 8-KiB allocation becomes the transposed logical B view [value,row].
    Element* const rounded_vnew = u;
    for (int index = thread_idx; index < kScoreElements;
         index += kThreadCount) {
      int const row = index / kValueBlock;
      int const value = index % kValueBlock;
      rounded_vnew[index] = to_bf16(vn[index]);
    }
    __syncthreads();
    {
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
      ResidentMma::mma(
          accum,
          p, kChunk, 1,
          rounded_vnew, 1, kValueBlock,
          thread_idx, valid, kValueBlock, valid);
      __syncthreads();
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int row, int value, float causal) {
            params.output[vo_offset(
                params, work, chunk_begin + row, value_base + value)] =
                to_bf16(
                    params.scale * os[row * kValueBlock + value] +
                    params.scale * causal);
          },
          valid, kValueBlock);
    }
    __syncthreads();

    // K^T @ round_bf16(exp2(gamma_last-gamma) * Vnew).  The same U arena is
    // reused, now with logical B strides [value,row].
    float const gamma_last = shared.gamma[valid - 1];
    float const state_decay = exp2_gate(gamma_last);
    for (int index = thread_idx; index < kScoreElements;
         index += kThreadCount) {
      int const row = index / kValueBlock;
      int const value = index % kValueBlock;
      rounded_vnew[index] = to_bf16(
          row < valid
              ? exp2_gate(gamma_last - shared.gamma[row]) * vn[index]
              : 0.0f);
    }
    __syncthreads();
#pragma unroll
    for (int feature_base = 0; feature_base < kHeadK;
         feature_base += kChunk) {
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
      Element const* const k =
          params.k + qk_offset(
                         params, work, chunk_begin, feature_base);
      ResidentMma::mma(
          accum,
          k, 1, qk_row_pitch,
          rounded_vnew, 1, kValueBlock,
          thread_idx, kChunk, kValueBlock, valid);
      __syncthreads();
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int feature, int value, float update) {
            int const h_index =
                (feature_base + feature) * kStateStride + value;
            shared.state[h_index] =
                state_decay * shared.state[h_index] + update;
          });
      __syncthreads();
    }
  }

  QZ_PPU_GDN_DEVICE static void ppu0010_prepare_u(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, int value_base,
      PreparedValueChunk& dst, SharedStorageView shared, int thread_idx) {
    Element const* const a = inverse_bf16(shared);
    Element* const generated_operand =
        reinterpret_cast<Element*>(shared.phase + kOffsetOState);
    for (int index = thread_idx; index < kScoreElements;
         index += kThreadCount) {
      int const value = index / kChunk;
      int const row = index % kChunk;
      generated_operand[index] = to_bf16(
          row < valid
              ? shared.beta[row] *
                    to_float(params.v[vo_offset(
                        params, work, chunk_begin + row,
                        value_base + value)])
              : 0.0f);
    }
    __syncthreads();
    auto accum = ResidentMma::make_accumulator();
    ResidentMma::clear(accum);
    ResidentMma::mma(
        accum,
        a, kChunk, 1,
        generated_operand, kChunk, 1,
        thread_idx, valid, kValueBlock, kChunk);
    __syncthreads();
    ResidentMma::visit_output(
        accum, thread_idx,
        [&](int row, int value, float product) {
          dst.u[row * kValueBlock + value] = to_bf16(product);
        });
  }

  QZ_PPU_GDN_DEVICE static void ppu0010_recur_h(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, int value_base,
      PreparedValueChunk& value_chunk,
      SharedStorageView shared, int thread_idx) {
    Element const* const w = w_matrix(shared);
    Element* const h_operand =
        reinterpret_cast<Element*>(shared.phase + kOffsetVNew);
    Element* const rounded_vnew = u_tile(shared);
    std::int64_t const qk_row_pitch =
        std::int64_t(params.problem.num_qk_heads) * kHeadK;

    // Publish the exact rounded H boundary once.  WH below and the later O
    // kernel consume the same bits, so splitting the kernels cannot invent a
    // second, different materialization.
    for (int index = thread_idx; index < kHeadK * kValueBlock;
         index += kThreadCount) {
      int const block = index / kScoreElements;
      int const within = index % kScoreElements;
      int const value = within / kChunk;
      int const feature = within % kChunk;
      value_chunk.h_start[index] = to_bf16(
          shared.state[(block * kChunk + feature) * kStateStride + value]);
    }
    __syncthreads();

    // Vnew = U - W H_start.  This is part of the true recurrent dependency:
    // the subsequent K^T Vnew update determines the next chunk's H.
    {
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
#pragma unroll
      for (int feature_base = 0; feature_base < kHeadK;
           feature_base += kChunk) {
        Element const* const source =
            value_chunk.h_start + (feature_base / kChunk) * kScoreElements;
        for (int index = thread_idx; index < kScoreElements;
             index += kThreadCount) {
          h_operand[index] = source[index];
        }
        __syncthreads();
        ResidentMma::mma(
            accum,
            w + feature_base, kHeadK, 1,
            h_operand, kChunk, 1,
            thread_idx, valid, kValueBlock, kChunk);
        __syncthreads();
      }
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int row, int value, float product) {
            value_chunk.v_new[row * kValueBlock + value] =
                row < valid
                    ? to_float(value_chunk.u[row * kValueBlock + value]) - product
                    : 0.0f;
          });
    }
    __syncthreads();

    float const gamma_last = shared.gamma[valid - 1];
    float const state_decay = exp2_gate(gamma_last);
    for (int index = thread_idx; index < kScoreElements;
         index += kThreadCount) {
      int const row = index / kValueBlock;
      rounded_vnew[index] = to_bf16(
          row < valid
              ? exp2_gate(gamma_last - shared.gamma[row]) *
                    value_chunk.v_new[index]
              : 0.0f);
    }
    __syncthreads();
#pragma unroll
    for (int feature_base = 0; feature_base < kHeadK;
         feature_base += kChunk) {
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
      Element const* const k =
          params.k + qk_offset(params, work, chunk_begin, feature_base);
      ResidentMma::mma(
          accum,
          k, 1, qk_row_pitch,
          rounded_vnew, 1, kValueBlock,
          thread_idx, kChunk, kValueBlock, valid);
      __syncthreads();
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int feature, int value, float update) {
            int const h_index =
                (feature_base + feature) * kStateStride + value;
            shared.state[h_index] =
                state_decay * shared.state[h_index] + update;
          });
      __syncthreads();
    }
    (void)value_base;
  }

  QZ_PPU_GDN_DEVICE static void ppu0010_write_o(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, int value_base,
      PreparedValueChunk const& value_chunk,
      SharedStorageView shared, int thread_idx) {
    float* const qh = o_state(shared);
    Element const* const p = causal_score(shared);
    Element* const operand =
        reinterpret_cast<Element*>(shared.phase + kOffsetVNew);
    Element* const rounded_vnew = u_tile(shared);
    std::int64_t const qk_row_pitch =
        std::int64_t(params.problem.num_qk_heads) * kHeadK;

    {
      auto accum = ResidentMma::make_accumulator();
      ResidentMma::clear(accum);
#pragma unroll
      for (int feature_base = 0; feature_base < kHeadK;
           feature_base += kChunk) {
        Element const* const source =
            value_chunk.h_start + (feature_base / kChunk) * kScoreElements;
        for (int index = thread_idx; index < kScoreElements;
             index += kThreadCount) {
          operand[index] = source[index];
        }
        __syncthreads();
        Element const* const q =
            params.q + qk_offset(params, work, chunk_begin, feature_base);
        ResidentMma::mma(
            accum,
            q, qk_row_pitch, 1,
            operand, kChunk, 1,
            thread_idx, valid, kValueBlock, kChunk);
        __syncthreads();
      }
      ResidentMma::visit_output(
          accum, thread_idx,
          [&](int row, int value, float product) {
            qh[row * kValueBlock + value] =
                row < valid ? exp2_gate(shared.gamma[row]) * product : 0.0f;
          });
    }
    __syncthreads();

    for (int index = thread_idx; index < kScoreElements;
         index += kThreadCount) {
      rounded_vnew[index] = to_bf16(value_chunk.v_new[index]);
    }
    __syncthreads();
    auto accum = ResidentMma::make_accumulator();
    ResidentMma::clear(accum);
    ResidentMma::mma(
        accum,
        p, kChunk, 1,
        rounded_vnew, 1, kValueBlock,
        thread_idx, valid, kValueBlock, valid);
    __syncthreads();
    ResidentMma::visit_output(
        accum, thread_idx,
        [&](int row, int value, float causal) {
          params.output[vo_offset(
              params, work, chunk_begin + row, value_base + value)] =
              to_bf16(params.scale * qh[row * kValueBlock + value] +
                      params.scale * causal);
        },
        valid, kValueBlock);
  }
  #endif

  QZ_PPU_GDN_DEVICE static void compute_value_tile(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, int value_base,
      SharedStorageView shared, int thread_idx) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_compute_value_tile(
        params, work, chunk_begin, valid, value_base, shared, thread_idx);
#elif defined(__HGGC_ARCH__)
    (void)params;
    (void)work;
    (void)chunk_begin;
    (void)valid;
    (void)value_base;
    (void)shared;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("generated GDN MMA requires ppu0010");
#else
    Element const* const a = inverse_bf16(shared);
    Element const* const w = w_matrix(shared);
    Element* const u = u_tile(shared);
    float* const os = o_state(shared);
    Element const* const p = causal_score(shared);
    float* const vn = v_new(shared);

    for (int index = thread_idx; index < kValueTileElements; index += kThreadCount) {
      int const row = index / kValueBlock;
      int const v = index % kValueBlock;
      float sum = 0.0f;
      for (int j = 0; j < kChunk; ++j) {
        // beta*V is likewise rounded to V's dtype before A@V.
        Element const vb = to_bf16(
            j < valid
                ? shared.beta[j] * to_float(params.v[vo_offset(
                      params, work, chunk_begin + j, value_base + v)])
                : 0.0f);
        sum += to_float(a[row * kChunk + j]) * to_float(vb);
      }
      u[index] = to_bf16(sum);
    }
    __syncthreads();

    for (int index = thread_idx; index < kValueTileElements; index += kThreadCount) {
      int const row = index / kValueBlock;
      int const v = index % kValueBlock;
      float qh = 0.0f;
      float wh = 0.0f;
      if (row < valid) {
        for (int d = 0; d < kHeadK; ++d) {
          // The reference pipeline materializes the recurrent-state boundary
          // in BF16 before QH/WH. Preserve that rounding point even though the
          // canonical in-CTA state remains FP32 across chunks.
          float const h =
              to_float(to_bf16(shared.state[d * kStateStride + v]));
          qh += to_float(params.q[qk_offset(params, work, chunk_begin + row, d)]) * h;
          wh += to_float(w[row * kHeadK + d]) * h;
        }
      }
      os[index] = row < valid ? exp2_gate(shared.gamma[row]) * qh : 0.0f;
      vn[index] = row < valid ? to_float(u[index]) - wh : 0.0f;
    }
    __syncthreads();

    for (int index = thread_idx; index < kValueTileElements; index += kThreadCount) {
      int const row = index / kValueBlock;
      int const v = index % kValueBlock;
      if (row < valid) {
        float causal = 0.0f;
        for (int j = 0; j <= row; ++j) {
          causal += to_float(p[row * kChunk + j]) *
                    to_float(to_bf16(vn[j * kValueBlock + v]));
        }
        params.output[vo_offset(params, work, chunk_begin + row, value_base + v)] =
            to_bf16(params.scale * os[index] + params.scale * causal);
      }
    }
    __syncthreads();

    float const gamma_last = shared.gamma[valid - 1];
    float const state_decay = exp2_gate(gamma_last);
    for (int index = thread_idx; index < kHeadK * kValueBlock; index += kThreadCount) {
      int const d = index / kValueBlock;
      int const v = index % kValueBlock;
      float update = 0.0f;
      for (int row = 0; row < valid; ++row) {
        // Match the chunk recurrence's BF16 dot boundary: gate in FP32, then
        // round the scaled delta value once before K^T V_new.
        Element const scaled_v = to_bf16(
            exp2_gate(gamma_last - shared.gamma[row]) *
            vn[row * kValueBlock + v]);
        update += to_float(params.k[qk_offset(params, work, chunk_begin + row, d)]) *
                  to_float(scaled_v);
      }
      int const h_index = d * kStateStride + v;
      shared.state[h_index] = state_decay * shared.state[h_index] + update;
    }
    __syncthreads();
#endif
  }

  QZ_PPU_GDN_DEVICE static void prepare_u_stage(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, int value_base,
      PreparedValueChunk& dst, SharedStorageView shared, int thread_idx) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_prepare_u(
        params, work, chunk_begin, valid, value_base, dst, shared, thread_idx);
#elif defined(__HGGC_ARCH__)
    (void)params;
    (void)work;
    (void)chunk_begin;
    (void)valid;
    (void)value_base;
    (void)dst;
    (void)shared;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("four-stage GDN U requires ppu0010");
#else
    Element const* const a = inverse_bf16(shared);
    for (int index = thread_idx; index < kValueTileElements;
         index += kThreadCount) {
      int const row = index / kValueBlock;
      int const value = index % kValueBlock;
      float sum = 0.0f;
      for (int j = 0; j < kChunk; ++j) {
        Element const vb = to_bf16(
            j < valid
                ? shared.beta[j] *
                      to_float(params.v[vo_offset(
                          params, work, chunk_begin + j,
                          value_base + value)])
                : 0.0f);
        sum += to_float(a[row * kChunk + j]) * to_float(vb);
      }
      dst.u[index] = to_bf16(sum);
    }
    __syncthreads();
#endif
  }

  QZ_PPU_GDN_DEVICE static void recur_h_stage(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, int value_base,
      PreparedValueChunk& value_chunk,
      SharedStorageView shared, int thread_idx) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_recur_h(
        params, work, chunk_begin, valid, value_base,
        value_chunk, shared, thread_idx);
#elif defined(__HGGC_ARCH__)
    (void)params;
    (void)work;
    (void)chunk_begin;
    (void)valid;
    (void)value_base;
    (void)value_chunk;
    (void)shared;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("four-stage GDN H requires ppu0010");
#else
    Element const* const w = w_matrix(shared);
    for (int index = thread_idx; index < kHeadK * kValueBlock;
         index += kThreadCount) {
      int const block = index / kScoreElements;
      int const within = index % kScoreElements;
      int const value = within / kChunk;
      int const feature = within % kChunk;
      value_chunk.h_start[index] = to_bf16(
          shared.state[(block * kChunk + feature) * kStateStride + value]);
    }
    __syncthreads();

    for (int index = thread_idx; index < kValueTileElements;
         index += kThreadCount) {
      int const row = index / kValueBlock;
      int const value = index % kValueBlock;
      float wh = 0.0f;
      if (row < valid) {
        for (int d = 0; d < kHeadK; ++d) {
          int const h_index =
              (d / kChunk) * kScoreElements + value * kChunk + d % kChunk;
          wh += to_float(w[row * kHeadK + d]) *
                to_float(value_chunk.h_start[h_index]);
        }
      }
      value_chunk.v_new[index] =
          row < valid ? to_float(value_chunk.u[index]) - wh : 0.0f;
    }
    __syncthreads();

    float const gamma_last = shared.gamma[valid - 1];
    float const state_decay = exp2_gate(gamma_last);
    for (int index = thread_idx; index < kHeadK * kValueBlock;
         index += kThreadCount) {
      int const d = index / kValueBlock;
      int const value = index % kValueBlock;
      float update = 0.0f;
      for (int row = 0; row < valid; ++row) {
        Element const scaled_v = to_bf16(
            exp2_gate(gamma_last - shared.gamma[row]) *
            value_chunk.v_new[row * kValueBlock + value]);
        update +=
            to_float(params.k[qk_offset(params, work, chunk_begin + row, d)]) *
            to_float(scaled_v);
      }
      shared.state[index] = state_decay * shared.state[index] + update;
    }
    __syncthreads();
    (void)value_base;
#endif
  }

  QZ_PPU_GDN_DEVICE static void write_o_stage(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_begin, int valid, int value_base,
      PreparedValueChunk const& value_chunk,
      SharedStorageView shared, int thread_idx) {
#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 100
    ppu0010_write_o(
        params, work, chunk_begin, valid, value_base,
        value_chunk, shared, thread_idx);
#elif defined(__HGGC_ARCH__)
    (void)params;
    (void)work;
    (void)chunk_begin;
    (void)valid;
    (void)value_base;
    (void)value_chunk;
    (void)shared;
    (void)thread_idx;
    CUTE_INVALID_CONTROL_PATH("four-stage GDN O requires ppu0010");
#else
    Element const* const p = causal_score(shared);
    for (int index = thread_idx; index < kValueTileElements;
         index += kThreadCount) {
      int const row = index / kValueBlock;
      int const value = index % kValueBlock;
      if (row >= valid) continue;
      float qh = 0.0f;
      for (int d = 0; d < kHeadK; ++d) {
        int const h_index =
            (d / kChunk) * kScoreElements + value * kChunk + d % kChunk;
        qh += to_float(params.q[qk_offset(params, work, chunk_begin + row, d)]) *
              to_float(value_chunk.h_start[h_index]);
      }
      qh *= exp2_gate(shared.gamma[row]);
      float causal = 0.0f;
      for (int j = 0; j <= row; ++j) {
        causal += to_float(p[row * kChunk + j]) *
                  to_float(to_bf16(
                      value_chunk.v_new[j * kValueBlock + value]));
      }
      params.output[vo_offset(
          params, work, chunk_begin + row, value_base + value)] =
          to_bf16(params.scale * qh + params.scale * causal);
    }
    __syncthreads();
#endif
  }

 public:
  CUTLASS_HOST_DEVICE static constexpr std::int64_t prepared_chunk_index(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_idx) {
    return ((std::int64_t(work.sequence_idx) * params.problem.num_v_heads +
             work.v_head_idx) *
                work.chunk_count +
            chunk_idx);
  }

  CUTLASS_HOST_DEVICE static constexpr std::int64_t prepared_value_chunk_index(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      int chunk_idx) {
    return prepared_chunk_index(params, work, chunk_idx) * kValueBlocks +
           work.value_tile_idx;
  }

  QZ_PPU_GDN_DEVICE static void prepare_chunk(
      Params const& params, PpuChunkedGdnPrepareWorkTileInfo const& prepare,
      PreparedChunk* workspace, SharedStorageView shared) {
    int const thread_idx = int(threadIdx.x);
    if (!prepare.valid || workspace == nullptr) return;
    PpuChunkedGdnWorkTileInfo const& work = prepare.head;
    int const chunk_begin = prepare.chunk_idx * kChunk;
    int const valid = load_chunk_scalars(
        params, work, prepare.chunk_idx, shared, thread_idx);
    global_dot(
        params, work, chunk_begin, valid,
        detail::PpuChunkedGdnGlobalDotKind::kCausalQk,
        shared, thread_idx);
    global_dot(
        params, work, chunk_begin, valid,
        detail::PpuChunkedGdnGlobalDotKind::kStrictLowerKk,
        shared, thread_idx);
    solve_inverse(shared, thread_idx);
    compute_w(params, work, chunk_begin, valid, shared, thread_idx);

    PreparedChunk& dst = workspace[prepared_chunk_index(
        params, work, prepare.chunk_idx)];
    Element const* const a = inverse_bf16(shared);
    Element const* const w = w_matrix(shared);
    Element const* const p = causal_score(shared);
    for (int i = thread_idx; i < kScoreElements; i += kThreadCount) {
      dst.inverse[i] = a[i];
      dst.causal[i] = p[i];
    }
    for (int i = thread_idx; i < kWElements; i += kThreadCount) {
      dst.w[i] = w[i];
    }
  }

  QZ_PPU_GDN_DEVICE static void prepare_value_chunk(
      Params const& params,
      PpuChunkedGdnValueChunkWorkTileInfo const& prepare,
      PreparedChunk const* common_workspace,
      PreparedValueChunk* value_workspace,
      SharedStorageView shared) {
    int const thread_idx = int(threadIdx.x);
    if (!prepare.valid || common_workspace == nullptr ||
        value_workspace == nullptr) {
      return;
    }
    PpuChunkedGdnWorkTileInfo const& work = prepare.head;
    int const chunk_begin = prepare.chunk_idx * kChunk;
    int const valid = load_chunk_scalars(
        params, work, prepare.chunk_idx, shared, thread_idx);
    PreparedChunk const& source = common_workspace[
        prepared_chunk_index(params, work, prepare.chunk_idx)];
    Element* const a = inverse_bf16(shared);
    for (int i = thread_idx; i < kScoreElements; i += kThreadCount) {
      a[i] = source.inverse[i];
    }
    __syncthreads();
    PreparedValueChunk& dst = value_workspace[
        prepared_value_chunk_index(params, work, prepare.chunk_idx)];
    prepare_u_stage(
        params, work, chunk_begin, valid, work.value_begin,
        dst, shared, thread_idx);
  }

  QZ_PPU_GDN_DEVICE static void run_h_recurrence(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      PreparedChunk const* common_workspace,
      PreparedValueChunk* value_workspace,
      SharedStorageView shared) {
    int const thread_idx = int(threadIdx.x);
    if (common_workspace == nullptr || value_workspace == nullptr ||
        work.value_count != kValueBlock || work.value_begin < 0 ||
        work.value_begin + kValueBlock > kHeadV) {
      return;
    }
    load_state(params, work, shared, thread_idx);
    for (int chunk = 0; chunk < work.chunk_count; ++chunk) {
      int const chunk_begin = chunk * kChunk;
      int const valid = load_chunk_scalars(
          params, work, chunk, shared, thread_idx);
      PreparedChunk const& common = common_workspace[
          prepared_chunk_index(params, work, chunk)];
      Element* const w = w_matrix(shared);
      for (int i = thread_idx; i < kWElements; i += kThreadCount) {
        w[i] = common.w[i];
      }
      __syncthreads();
      PreparedValueChunk& value_chunk = value_workspace[
          prepared_value_chunk_index(params, work, chunk)];
      recur_h_stage(
          params, work, chunk_begin, valid, work.value_begin,
          value_chunk, shared, thread_idx);
    }
    store_final_state(params, work, shared, thread_idx);
  }

  QZ_PPU_GDN_DEVICE static void write_output_chunk(
      Params const& params,
      PpuChunkedGdnValueChunkWorkTileInfo const& output_work,
      PreparedChunk const* common_workspace,
      PreparedValueChunk const* value_workspace,
      SharedStorageView shared) {
    int const thread_idx = int(threadIdx.x);
    if (!output_work.valid || common_workspace == nullptr ||
        value_workspace == nullptr) {
      return;
    }
    PpuChunkedGdnWorkTileInfo const& work = output_work.head;
    int const chunk_begin = output_work.chunk_idx * kChunk;
    int const valid = load_chunk_scalars(
        params, work, output_work.chunk_idx, shared, thread_idx);
    PreparedChunk const& common = common_workspace[
        prepared_chunk_index(params, work, output_work.chunk_idx)];
    Element* const p = causal_score(shared);
    for (int i = thread_idx; i < kScoreElements; i += kThreadCount) {
      p[i] = common.causal[i];
    }
    __syncthreads();
    PreparedValueChunk const& value_chunk = value_workspace[
        prepared_value_chunk_index(params, work, output_work.chunk_idx)];
    write_o_stage(
        params, work, chunk_begin, valid, work.value_begin,
        value_chunk, shared, thread_idx);
  }

  QZ_PPU_GDN_DEVICE static void run_prepared(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      PreparedChunk const* workspace, SharedStorageView shared) {
    int const thread_idx = int(threadIdx.x);
    if (workspace == nullptr || work.value_count != kValueBlock ||
        work.value_begin < 0 || work.value_begin + kValueBlock > kHeadV) {
      return;
    }
    load_state(params, work, shared, thread_idx);
    for (int chunk = 0; chunk < work.chunk_count; ++chunk) {
      int const chunk_begin = chunk * kChunk;
      int const valid = load_chunk_scalars(
          params, work, chunk, shared, thread_idx);
      PreparedChunk const& src = workspace[
          prepared_chunk_index(params, work, chunk)];
      Element* const a = inverse_bf16(shared);
      Element* const w = w_matrix(shared);
      Element* const p = causal_score(shared);
      for (int i = thread_idx; i < kScoreElements; i += kThreadCount) {
        a[i] = src.inverse[i];
        p[i] = src.causal[i];
      }
      for (int i = thread_idx; i < kWElements; i += kThreadCount) {
        w[i] = src.w[i];
      }
      __syncthreads();
      compute_value_tile(
          params, work, chunk_begin, valid, work.value_begin,
          shared, thread_idx);
    }
    store_final_state(params, work, shared, thread_idx);
  }

  QZ_PPU_GDN_DEVICE static void run(
      Params const& params, PpuChunkedGdnWorkTileInfo const& work,
      SharedStorageView shared) {
    int const thread_idx = int(threadIdx.x);
    // Scheduler/collective mismatch must fail at compile time in the shipping
    // type.  These uniform runtime values document the address contract used
    // below; the scheduler exhaustiveness gate proves every V column has one
    // and only one owner.
    if (work.value_count != kValueBlock ||
        work.value_begin < 0 || work.value_begin + kValueBlock > kHeadV) {
      return;
    }
    load_state(params, work, shared, thread_idx);
    for (int chunk = 0; chunk < work.chunk_count; ++chunk) {
      int const chunk_begin = chunk * kChunk;
      int const valid = load_chunk_scalars(params, work, chunk, shared, thread_idx);

      // QK runs first so its compact BF16 causal score can live above the
      // 32-KiB AIU arena while KK reuses that arena.
      global_dot(
          params, work, chunk_begin, valid,
          detail::PpuChunkedGdnGlobalDotKind::kCausalQk,
          shared, thread_idx);
      global_dot(
          params, work, chunk_begin, valid,
          detail::PpuChunkedGdnGlobalDotKind::kStrictLowerKk,
          shared, thread_idx);
      solve_inverse(shared, thread_idx);
      compute_w(params, work, chunk_begin, valid, shared, thread_idx);
      compute_value_tile(
          params, work, chunk_begin, valid, work.value_begin,
          shared, thread_idx);
    }
    store_final_state(params, work, shared, thread_idx);
  }
};

}  // namespace cutlass::linear_attention

#undef QZ_PPU_GDN_DEVICE
