/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Register-resident C64xC64xK64 BF16/BF16->FP32 MMA delivery for chunked GDN.
 *
 * This helper deliberately borrows the exact production TiledMma instead of
 * spelling a second lane map.  partition_A/B(identity) owns the logical source
 * coordinates, partition_fragment_A/B owns the physical register ABI, and
 * partition_C(identity) owns the unique output coordinates.  Consequently a
 * generated operand can enter the PPU MMA without inventing a
 * register-to-swizzled-shared writer.
 *
 * The helper is visible only to hgcc builds.  NVIDIA nvcc keeps compiling the
 * independent scalar GDN reference and must not acquire a fake PPU execution
 * path merely by including this header.
 **************************************************************************************************/
#pragma once

#include <cstdint>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"

#if defined(__HGGCCC__)
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor.hpp"
#endif

namespace cutlass::linear_attention::detail {

#if defined(__HGGCCC__)

// Local ownership oracles instantiate the __HGGCCC__ type branch with NVIDIA
// nvcc.  That compiler cannot emit actlize's PPU CuTe partition functions as
// device code (its fake-PPU pass diagnoses host-only `_`/`product` constants),
// while hgcc can and must.  Keep the exact same function body host-only in the
// oracle compiler; production hgcc receives the promised host/device surface.
#if defined(__NVCC__)
#define QZ_PPU_GDN_COORD_HOST_DEVICE inline
#else
#define QZ_PPU_GDN_COORD_HOST_DEVICE CUTLASS_HOST_DEVICE
#endif

template <class TiledMma_>
struct PpuChunkedGdnResidentMmaBf16C64K64 {
  using TiledMma = TiledMma_;
  using Element = cutlass::bfloat16_t;

  static constexpr int kM = 64;
  static constexpr int kN = 64;
  static constexpr int kK = 64;
  static constexpr int kThreads = 128;
  static constexpr int kOperandValuesPerThread = 64;
  static constexpr int kAccumulatorValuesPerThread = 32;

  using TileShape = cute::Shape<cute::Int<kM>, cute::Int<kN>, cute::Int<kK>>;
  using OperandShape = cute::Shape<cute::Int<kM>, cute::Int<kK>>;
  using OutputShape = cute::Shape<cute::Int<kM>, cute::Int<kN>>;
  using Accumulator = decltype(cute::partition_fragment_C(
      TiledMma{}, OutputShape{}));

  static_assert(cute::size(TiledMma{}) == kThreads,
                "resident C64 GDN MMA requires the production four-warp TiledMma");
  static_assert(cute::size<0>(typename TiledMma::AtomShape_MNK{}) == 16 &&
                    cute::size<1>(typename TiledMma::AtomShape_MNK{}) == 16 &&
                    cute::size<2>(typename TiledMma::AtomShape_MNK{}) == 16,
                "resident C64 GDN MMA requires the PPU m16n16k16 atom");

  CUTLASS_HOST_DEVICE static Accumulator make_accumulator() {
    return cute::partition_fragment_C(TiledMma{}, OutputShape{});
  }

  CUTLASS_HOST_DEVICE static void clear(Accumulator& accum) {
    cute::clear(accum);
  }

  // These three iterators are both the production delivery mechanism and the
  // host-oracle surface.  A local gate can exhaust the exact maps used below
  // without copying lane arithmetic into a parallel model.
  template <class Visitor>
  QZ_PPU_GDN_COORD_HOST_DEVICE static void for_each_a_coordinate(
      int thread_idx, Visitor&& visitor) {
    using namespace cute;
    auto identity = make_identity_tensor(OperandShape{});
    auto coord = TiledMma{}.get_thread_slice(thread_idx).partition_A(identity);
    static_assert(decltype(size(coord))::value == kOperandValuesPerThread,
                  "production A coordinate ownership changed");
#pragma unroll
    for (int slot = 0; slot < int(size(coord)); ++slot) {
      auto const logical = coord(slot);
      visitor(slot, int(get<0>(logical)), int(get<1>(logical)));
    }
  }

  template <class Visitor>
  QZ_PPU_GDN_COORD_HOST_DEVICE static void for_each_b_coordinate(
      int thread_idx, Visitor&& visitor) {
    using namespace cute;
    auto identity = make_identity_tensor(OperandShape{});
    auto coord = TiledMma{}.get_thread_slice(thread_idx).partition_B(identity);
    static_assert(decltype(size(coord))::value == kOperandValuesPerThread,
                  "production B coordinate ownership changed");
#pragma unroll
    for (int slot = 0; slot < int(size(coord)); ++slot) {
      auto const logical = coord(slot);
      visitor(slot, int(get<0>(logical)), int(get<1>(logical)));
    }
  }

  template <class Visitor>
  QZ_PPU_GDN_COORD_HOST_DEVICE static void for_each_c_coordinate(
      int thread_idx, Visitor&& visitor) {
    using namespace cute;
    auto identity = make_identity_tensor(OutputShape{});
    auto coord = TiledMma{}.get_thread_slice(thread_idx).partition_C(identity);
    static_assert(decltype(size(coord))::value == kAccumulatorValuesPerThread,
                  "production C coordinate ownership changed");
#pragma unroll
    for (int slot = 0; slot < int(size(coord)); ++slot) {
      auto const logical = coord(slot);
      visitor(slot, int(get<0>(logical)), int(get<1>(logical)));
    }
  }

  // Compute accum += A[M,K] * B[N,K].  Both operands are logical matrices;
  // their physical memory layout is expressed solely by the two element
  // strides.  This covers row-major generated operands as well as K^T views
  // without a second TiledMma or a hidden transpose convention.
  //
  // valid_* provide exact zero fill for the final chunk.  All 128 CTA threads
  // must call this method with the same bounds and pointers.
  CUTLASS_DEVICE static void mma(
      Accumulator& accum,
      Element const* operand_a,
      std::int64_t stride_a_m,
      std::int64_t stride_a_k,
      Element const* operand_b,
      std::int64_t stride_b_n,
      std::int64_t stride_b_k,
      int thread_idx,
      int valid_m = kM,
      int valid_n = kN,
      int valid_k = kK) {
#if defined(__NVCC__)
    // nvcc is used only to instantiate the PPU type/ownership oracle.  It
    // cannot compile actlize's PPU CuTe device partitions; the independent
    // NVIDIA scalar GDN path never calls this method.
    (void)accum;
    (void)operand_a;
    (void)stride_a_m;
    (void)stride_a_k;
    (void)operand_b;
    (void)stride_b_n;
    (void)stride_b_k;
    (void)thread_idx;
    (void)valid_m;
    (void)valid_n;
    (void)valid_k;
#else
    using namespace cute;

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);
    // partition_fragment_* needs an ordinary strided tensor to derive a
    // compact register layout.  The tensor below is a shape/layout donor only:
    // loads still use the caller's explicit logical strides and the identity
    // partitions above remain the coordinate authority.
    auto fragment_shape_a = make_tensor(
        make_gmem_ptr(operand_a),
        make_layout(OperandShape{},
                    Stride<Int<kK>, Int<1>>{}));
    auto fragment_shape_b = make_tensor(
        make_gmem_ptr(operand_b),
        make_layout(OperandShape{},
                    Stride<Int<kK>, Int<1>>{}));
    auto fragment_a = thread_mma.partition_fragment_A(fragment_shape_a);
    auto fragment_b = thread_mma.partition_fragment_B(fragment_shape_b);

    static_assert(decltype(size(fragment_a))::value ==
                      kOperandValuesPerThread,
                  "production A fragment ownership changed");
    static_assert(decltype(size(fragment_b))::value ==
                      kOperandValuesPerThread,
                  "production B fragment ownership changed");
    static_assert(decltype(size(Accumulator{}))::value ==
                      kAccumulatorValuesPerThread,
                  "production C fragment ownership changed");

    for_each_a_coordinate(
        thread_idx, [&](int slot, int row, int reduction) {
          if (row < valid_m && reduction < valid_k) {
            fragment_a(slot) =
                operand_a[std::int64_t(row) * stride_a_m +
                          std::int64_t(reduction) * stride_a_k];
          } else {
            fragment_a(slot) = Element{};
          }
        });

    for_each_b_coordinate(
        thread_idx, [&](int slot, int column, int reduction) {
          if (column < valid_n && reduction < valid_k) {
            fragment_b(slot) =
                operand_b[std::int64_t(column) * stride_b_n +
                          std::int64_t(reduction) * stride_b_k];
          } else {
            fragment_b(slot) = Element{};
          }
        });

    // The three-tensor overload means accum = A*B + accum.  CuTe expands the
    // four K atoms carried by the production register fragments.
    cute::gemm(tiled_mma, fragment_a, fragment_b, accum);
#endif
  }

  // Invoke visitor(row, column, value) exactly once for every logical C
  // coordinate across the CTA.  No caller may infer output ownership from
  // lane arithmetic; the production partition_C map is the sole authority.
  template <class Visitor>
  CUTLASS_DEVICE static void visit_output(
      Accumulator const& accum, int thread_idx, Visitor&& visitor,
      int valid_m = kM, int valid_n = kN) {
#if defined(__NVCC__)
    (void)accum;
    (void)thread_idx;
    (void)visitor;
    (void)valid_m;
    (void)valid_n;
#else
    for_each_c_coordinate(
        thread_idx, [&](int slot, int row, int column) {
          if (row < valid_m && column < valid_n) {
            visitor(row, column, accum(slot));
          }
        });
#endif
  }

  template <class Output>
  CUTLASS_DEVICE static void store(
      Accumulator const& accum,
      Output* output,
      std::int64_t stride_m,
      std::int64_t stride_n,
      int thread_idx,
      int valid_m = kM,
      int valid_n = kN) {
#if defined(__NVCC__)
    (void)accum;
    (void)output;
    (void)stride_m;
    (void)stride_n;
    (void)thread_idx;
    (void)valid_m;
    (void)valid_n;
#else
    for_each_c_coordinate(
        thread_idx, [&](int slot, int row, int column) {
          if (row < valid_m && column < valid_n) {
            output[std::int64_t(row) * stride_m +
                   std::int64_t(column) * stride_n] = Output(accum(slot));
          }
        });
#endif
  }
};

#undef QZ_PPU_GDN_COORD_HOST_DEVICE

#endif  // defined(__HGGCCC__)

}  // namespace cutlass::linear_attention::detail
