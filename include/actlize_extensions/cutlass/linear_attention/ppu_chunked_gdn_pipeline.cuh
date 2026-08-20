/***************************************************************************************************
 * Copyright (c) 2026 quactlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Two-stage PPU chunked-GDN forward pipeline.  The prepare kernel materializes
 * the BF16 A/W/P seam once per logical (sequence,V-head,chunk); two independent
 * BV64 recurrence CTAs then consume that seam without recomputing QK, KK,
 * inverse, or W.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_collective.cuh"

#if defined(__CUDACC__) && !defined(__HGGCCC__)
#define QZ_PPU_GDN_PIPELINE_DEVICE __device__ __forceinline__
#else
#define QZ_PPU_GDN_PIPELINE_DEVICE CUTLASS_DEVICE
#endif

namespace cutlass::linear_attention {

template <class Arguments_, class Traits_>
struct PpuChunkedGdnTwoStagePipeline {
  using Arguments = Arguments_;
  using Traits = Traits_;
  using Collective =
      PpuChunkedGdnCollectiveBf16C64D128BV64<Traits, Arguments>;
  using PreparedChunk = typename Collective::PreparedChunk;
  using PrepareScheduler = PpuChunkedGdnPrepareScheduler<Traits>;
  using RecurrenceScheduler =
      PpuChunkedGdnScheduler<Traits, Collective::kValueBlock>;

  struct Params {
    Arguments args{};
    PreparedChunk* workspace = nullptr;
  };

  static_assert(std::is_trivially_copyable_v<Params>,
                "two-stage params must cross the kernel ABI by value");
  static_assert(Collective::kPreparedChunkBytes == 32768,
                "pipeline workspace seam changed");

  static constexpr std::size_t get_workspace_size(
      PpuChunkedGdnProblem const& problem) {
    std::int32_t const cells = PrepareScheduler::grid_size(problem);
    if (cells <= 0 ||
        std::size_t(cells) >
            std::numeric_limits<std::size_t>::max() / sizeof(PreparedChunk)) {
      return 0;
    }
    return std::size_t(cells) * sizeof(PreparedChunk);
  }

  static PpuChunkedGdnStatus argument_status(
      Arguments const& args, void* workspace, std::size_t workspace_bytes) {
    PpuChunkedGdnStatus const base = Collective::argument_status(args);
    if (base != PpuChunkedGdnStatus::kSuccess) return base;
    std::size_t const required = get_workspace_size(args.problem);
    if (required == 0 || workspace == nullptr || workspace_bytes < required ||
        reinterpret_cast<std::uintptr_t>(workspace) % alignof(PreparedChunk) != 0) {
      return PpuChunkedGdnStatus::kInsufficientWorkspace;
    }
    return PpuChunkedGdnStatus::kSuccess;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{args, reinterpret_cast<PreparedChunk*>(workspace)};
  }
};

template <class Pipeline_>
struct PpuChunkedGdnPrepareKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::PrepareScheduler;
  using SharedStorage = typename Collective::SharedStorage;

  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));

  static dim3 get_grid_shape(Params const& params) {
    return dim3(
        static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  QZ_PPU_GDN_PIPELINE_DEVICE void operator()(
      Params const& params, char* smem_buf) {
    PpuChunkedGdnPrepareWorkTileInfo const work =
        Scheduler::work(static_cast<int>(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::prepare_chunk(params.args, work, params.workspace, shared);
  }
};

template <class Pipeline_>
struct PpuChunkedGdnRecurrenceKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::RecurrenceScheduler;
  using SharedStorage = typename Collective::SharedStorage;

  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));

  static dim3 get_grid_shape(Params const& params) {
    return dim3(
        static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  QZ_PPU_GDN_PIPELINE_DEVICE void operator()(
      Params const& params, char* smem_buf) {
    PpuChunkedGdnWorkTileInfo const work =
        Scheduler::work(static_cast<int>(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    SharedStorage& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::run_prepared(params.args, work, params.workspace, shared);
  }
};

// Four-stage PPU forward DAG:
//   1. common A/W/P prepare              [sequence, V-head, chunk]
//   2. U = A @ (beta V)                 [sequence, V-head, chunk, BV64]
//   3. serial H recurrence (WH + K^T dV)[sequence, V-head, BV64]
//   4. O = exp(gamma) QH + P dV         [sequence, V-head, chunk, BV64]
// Only stage 3 owns a cross-chunk dependency.  The other three stages expose
// every independent chunk to the device grid.
template <class Arguments_, class Traits_>
struct PpuChunkedGdnFourStagePipeline {
  using Arguments = Arguments_;
  using Traits = Traits_;
  using Collective =
      PpuChunkedGdnCollectiveBf16C64D128BV64<Traits, Arguments>;
  using PreparedChunk = typename Collective::PreparedChunk;
  using PreparedValueChunk = typename Collective::PreparedValueChunk;
  using PrepareScheduler = PpuChunkedGdnPrepareScheduler<Traits>;
  using ValueScheduler =
      PpuChunkedGdnValueChunkScheduler<Traits, Collective::kValueBlock>;
  using RecurrenceScheduler =
      PpuChunkedGdnScheduler<Traits, Collective::kValueBlock>;

  struct Params {
    Arguments args{};
    PreparedChunk* common_workspace = nullptr;
    PreparedValueChunk* value_workspace = nullptr;
  };

  static_assert(std::is_trivially_copyable_v<Params>,
                "four-stage params must cross the kernel ABI by value");
  static_assert(Collective::kPreparedChunkBytes == 32768 &&
                    Collective::kPreparedValueChunkBytes == 40960,
                "four-stage workspace seams changed");

  static constexpr std::size_t common_workspace_size(
      PpuChunkedGdnProblem const& problem) {
    std::int32_t const cells = PrepareScheduler::grid_size(problem);
    if (cells <= 0 ||
        std::size_t(cells) >
            std::numeric_limits<std::size_t>::max() / sizeof(PreparedChunk)) {
      return 0;
    }
    return std::size_t(cells) * sizeof(PreparedChunk);
  }

  static constexpr std::size_t value_workspace_size(
      PpuChunkedGdnProblem const& problem) {
    std::int32_t const cells = ValueScheduler::grid_size(problem);
    if (cells <= 0 ||
        std::size_t(cells) >
            std::numeric_limits<std::size_t>::max() /
                sizeof(PreparedValueChunk)) {
      return 0;
    }
    return std::size_t(cells) * sizeof(PreparedValueChunk);
  }

  static constexpr std::size_t get_workspace_size(
      PpuChunkedGdnProblem const& problem) {
    std::size_t const common = common_workspace_size(problem);
    std::size_t const value = value_workspace_size(problem);
    return common == 0 || value == 0 ||
                   common > std::numeric_limits<std::size_t>::max() - value
               ? 0
               : common + value;
  }

  static PpuChunkedGdnStatus argument_status(
      Arguments const& args, void* workspace, std::size_t workspace_bytes) {
    PpuChunkedGdnStatus const base = Collective::argument_status(args);
    if (base != PpuChunkedGdnStatus::kSuccess) return base;
    std::size_t const required = get_workspace_size(args.problem);
    if (required == 0 || workspace == nullptr || workspace_bytes < required ||
        reinterpret_cast<std::uintptr_t>(workspace) % alignof(PreparedChunk) != 0) {
      return PpuChunkedGdnStatus::kInsufficientWorkspace;
    }
    return PpuChunkedGdnStatus::kSuccess;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    auto* const bytes = reinterpret_cast<std::uint8_t*>(workspace);
    auto* const common = reinterpret_cast<PreparedChunk*>(bytes);
    auto* const values = reinterpret_cast<PreparedValueChunk*>(
        bytes + common_workspace_size(args.problem));
    return Params{args, common, values};
  }
};

template <class Pipeline_>
struct PpuChunkedGdnFourStagePrepareKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::PrepareScheduler;
  using SharedStorage = typename Collective::SharedStorage;
  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static dim3 get_grid_shape(Params const& params) {
    return dim3(static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }
  QZ_PPU_GDN_PIPELINE_DEVICE void operator()(Params const& params, char* smem_buf) {
    auto const work = Scheduler::work(int(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::prepare_chunk(params.args, work, params.common_workspace, shared);
  }
};

template <class Pipeline_>
struct PpuChunkedGdnUKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::ValueScheduler;
  using SharedStorage = typename Collective::SharedStorage;
  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static dim3 get_grid_shape(Params const& params) {
    return dim3(static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }
  QZ_PPU_GDN_PIPELINE_DEVICE void operator()(Params const& params, char* smem_buf) {
    auto const work = Scheduler::work(int(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::prepare_value_chunk(
        params.args, work, params.common_workspace,
        params.value_workspace, shared);
  }
};

template <class Pipeline_>
struct PpuChunkedGdnHKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::RecurrenceScheduler;
  using SharedStorage = typename Collective::SharedStorage;
  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static dim3 get_grid_shape(Params const& params) {
    return dim3(static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }
  QZ_PPU_GDN_PIPELINE_DEVICE void operator()(Params const& params, char* smem_buf) {
    auto const work = Scheduler::work(int(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::run_h_recurrence(
        params.args, work, params.common_workspace,
        params.value_workspace, shared);
  }
};

template <class Pipeline_>
struct PpuChunkedGdnOKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::ValueScheduler;
  using SharedStorage = typename Collective::SharedStorage;
  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static dim3 get_grid_shape(Params const& params) {
    return dim3(static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }
  QZ_PPU_GDN_PIPELINE_DEVICE void operator()(Params const& params, char* smem_buf) {
    auto const work = Scheduler::work(int(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::write_output_chunk(
        params.args, work, params.common_workspace,
        params.value_workspace, shared);
  }
};

}  // namespace cutlass::linear_attention

#undef QZ_PPU_GDN_PIPELINE_DEVICE
