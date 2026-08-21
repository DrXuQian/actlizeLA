/***************************************************************************************************
 * Copyright (c) 2026 actlize contributors.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Four-launch post-cumsum forward DAG aligned to flash-linear-attention:
 *   KKT/solve -> recompute W+U -> recurrent H -> output O.
 **************************************************************************************************/
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_triton_collective.cuh"

#if defined(__CUDACC__) && !defined(__HGGCCC__)
#define ACTLIZE_GDN_TRITON_PIPELINE_DEVICE __device__ __forceinline__
#else
#define ACTLIZE_GDN_TRITON_PIPELINE_DEVICE CUTLASS_DEVICE
#endif

namespace cutlass::linear_attention {

template <class Arguments_, class Traits_>
struct PpuChunkedGdnTritonPipeline {
  using Arguments = Arguments_;
  using Traits = Traits_;
  using Collective =
      PpuChunkedGdnTritonCollectiveBf16C64D128BV64<Traits, Arguments>;
  using ChunkWorkspace = typename Collective::ChunkWorkspace;
  using PrepareScheduler = PpuChunkedGdnPrepareScheduler<Traits>;
  using RecurrenceScheduler =
      PpuChunkedGdnScheduler<Traits, Collective::kValueBlock>;
  using ValueScheduler =
      PpuChunkedGdnValueChunkScheduler<Traits, Collective::kValueBlock>;

  struct Params {
    Arguments args{};
    ChunkWorkspace* workspace = nullptr;
  };

  static_assert(std::is_trivially_copyable_v<Params>,
                "Triton-aligned params must cross the kernel ABI by value");
  static_assert(Collective::kChunkWorkspaceBytes == 90112,
                "Triton-aligned global seam changed");

  static constexpr std::size_t get_workspace_size(
      PpuChunkedGdnProblem const& problem) {
    std::int32_t const cells = PrepareScheduler::grid_size(problem);
    if (cells <= 0 ||
        std::size_t(cells) >
            std::numeric_limits<std::size_t>::max() / sizeof(ChunkWorkspace)) {
      return 0;
    }
    return std::size_t(cells) * sizeof(ChunkWorkspace);
  }

  static PpuChunkedGdnStatus argument_status(
      Arguments const& args, void* workspace, std::size_t workspace_bytes) {
    PpuChunkedGdnStatus const base = Collective::argument_status(args);
    if (base != PpuChunkedGdnStatus::kSuccess) return base;
    std::size_t const required = get_workspace_size(args.problem);
    if (required == 0 || workspace == nullptr || workspace_bytes < required ||
        reinterpret_cast<std::uintptr_t>(workspace) % alignof(ChunkWorkspace) != 0) {
      return PpuChunkedGdnStatus::kInsufficientWorkspace;
    }
    return PpuChunkedGdnStatus::kSuccess;
  }

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return Params{args, reinterpret_cast<ChunkWorkspace*>(workspace)};
  }
};

template <class Pipeline_>
struct PpuChunkedGdnTritonKktKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::PrepareScheduler;
  using SharedStorage = typename Collective::KktSharedStorage;
  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static dim3 get_grid_shape(Params const& params) {
    return dim3(static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }
  ACTLIZE_GDN_TRITON_PIPELINE_DEVICE void operator()(
      Params const& params, char* smem_buf) {
    auto const work = Scheduler::work(int(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::run_kkt_solve(params.args, work, params.workspace, shared);
  }
};

template <class Pipeline_>
struct PpuChunkedGdnTritonWuKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::PrepareScheduler;
  using SharedStorage = typename Collective::WuSharedStorage;
  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static dim3 get_grid_shape(Params const& params) {
    return dim3(static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }
  ACTLIZE_GDN_TRITON_PIPELINE_DEVICE void operator()(
      Params const& params, char* smem_buf) {
    auto const work = Scheduler::work(int(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::run_wu(params.args, work, params.workspace, shared);
  }
};

template <class Pipeline_>
struct PpuChunkedGdnTritonHKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::RecurrenceScheduler;
  using SharedStorage = typename Collective::HSharedStorage;
  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static dim3 get_grid_shape(Params const& params) {
    return dim3(static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }
  ACTLIZE_GDN_TRITON_PIPELINE_DEVICE void operator()(
      Params const& params, char* smem_buf) {
    auto const work = Scheduler::work(int(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::run_h(params.args, work, params.workspace, shared);
  }
};

template <class Pipeline_>
struct PpuChunkedGdnTritonOKernel {
  using Pipeline = Pipeline_;
  using Params = typename Pipeline::Params;
  using Collective = typename Pipeline::Collective;
  using Scheduler = typename Pipeline::ValueScheduler;
  using SharedStorage = typename Collective::OSharedStorage;
  static constexpr std::uint32_t MaxThreadsPerBlock = Collective::kThreadCount;
  static constexpr std::uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr int SharedStorageSize = int(sizeof(SharedStorage));
  static dim3 get_grid_shape(Params const& params) {
    return dim3(static_cast<unsigned>(Scheduler::grid_size(params.args.problem)), 1, 1);
  }
  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }
  ACTLIZE_GDN_TRITON_PIPELINE_DEVICE void operator()(
      Params const& params, char* smem_buf) {
    auto const work = Scheduler::work(int(blockIdx.x), params.args.problem);
    if (!work.valid) return;
    auto& shared = *reinterpret_cast<SharedStorage*>(smem_buf);
    Collective::run_o(params.args, work, params.workspace, shared);
  }
};

}  // namespace cutlass::linear_attention

#undef ACTLIZE_GDN_TRITON_PIPELINE_DEVICE
