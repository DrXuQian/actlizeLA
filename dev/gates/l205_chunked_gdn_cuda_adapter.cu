// Local RTX correctness adapter for the exact production collective body.
//
// RTX 5090 exposes 101376 bytes of opt-in shared memory per block while the
// PPU split-V specialization owns 107008 bytes.  A normal CUDA launch therefore
// cannot instantiate the exact body.  This test-only adapter places the
// unchanged SharedStorage object in global memory and passes it to the same
// PpuChunkedGdnKernel::operator().  Every scalar CUDA fallback instruction,
// synchronization point and BF16 materialization is unchanged; only the
// address space used by the scratch ledger differs.  Performance is out of
// scope.  The PPU ABI arm separately launches the shipping shared-memory path.

#include <cstdint>

#include <cuda_runtime.h>

#include "cutlass/bfloat16.h"
#include "quactlize_ppu_linear_attention.h"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_kernel.cuh"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_pipeline.cuh"

namespace {

using Element = cutlass::bfloat16_t;
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
using Arguments =
    cutlass::linear_attention::PpuChunkedGdnArguments<Element, Element, float>;
using Kernel = cutlass::linear_attention::PpuChunkedGdnKernel<Arguments, Traits>;
using Pipeline =
    cutlass::linear_attention::PpuChunkedGdnTwoStagePipeline<Arguments, Traits>;
using PrepareKernel = cutlass::linear_attention::PpuChunkedGdnPrepareKernel<Pipeline>;
using RecurrenceKernel =
    cutlass::linear_attention::PpuChunkedGdnRecurrenceKernel<Pipeline>;
using FourStagePipeline =
    cutlass::linear_attention::PpuChunkedGdnFourStagePipeline<Arguments, Traits>;
using FourPrepareKernel =
    cutlass::linear_attention::PpuChunkedGdnFourStagePrepareKernel<FourStagePipeline>;
using UKernel = cutlass::linear_attention::PpuChunkedGdnUKernel<FourStagePipeline>;
using HKernel = cutlass::linear_attention::PpuChunkedGdnHKernel<FourStagePipeline>;
using OKernel = cutlass::linear_attention::PpuChunkedGdnOKernel<FourStagePipeline>;
constexpr std::size_t kFourStageScratchStride =
    sizeof(typename HKernel::SharedStorage);

__global__ void chunked_gdn_global_scratch_kernel(
    Arguments args, typename Kernel::SharedStorage* scratch) {
  Kernel kernel;
  kernel(args, reinterpret_cast<char*>(&scratch[blockIdx.x]));
}

__global__ void chunked_gdn_prepare_global_scratch_kernel(
    typename Pipeline::Params params,
    typename PrepareKernel::SharedStorage* scratch) {
  PrepareKernel{}(params, reinterpret_cast<char*>(&scratch[blockIdx.x]));
}

__global__ void chunked_gdn_recurrence_global_scratch_kernel(
    typename Pipeline::Params params,
    typename RecurrenceKernel::SharedStorage* scratch) {
  RecurrenceKernel{}(params, reinterpret_cast<char*>(&scratch[blockIdx.x]));
}

template <class DeviceKernel>
__global__ void chunked_gdn_four_stage_global_scratch_kernel(
    typename FourStagePipeline::Params params,
    std::uint8_t* scratch) {
  DeviceKernel{}(
      params,
      reinterpret_cast<char*>(scratch + blockIdx.x * kFourStageScratchStride));
}

Arguments make_arguments(
    std::uint16_t const* q, std::uint16_t const* k, std::uint16_t const* v,
    float const* gamma_log2_cumsum, float const* beta,
    float const* initial_state, std::uint16_t* output, float* final_state,
    quactlize_ppu_chunked_gdn_problem_v1 const& problem, float scale) {
  Arguments args{};
  args.q = reinterpret_cast<Element const*>(q);
  args.k = reinterpret_cast<Element const*>(k);
  args.v = reinterpret_cast<Element const*>(v);
  args.gamma_log2_cumsum = gamma_log2_cumsum;
  args.beta = beta;
  args.initial_state = initial_state;
  args.output = reinterpret_cast<Element*>(output);
  args.final_state = final_state;
  args.problem = {
      problem.total_tokens,
      problem.num_sequences,
      problem.sequence_length,
      problem.num_qk_heads,
      problem.num_v_heads,
      problem.head_size_k,
      problem.head_size_v,
      problem.chunk_size,
  };
  args.scale = scale;
  return args;
}

}  // namespace

extern "C" int quactlize_ppu_chunked_gdn_fwd_bf16_v1(
    std::uint16_t const* q,
    std::uint16_t const* k,
    std::uint16_t const* v,
    float const* gamma_log2_cumsum,
    float const* beta,
    float const* initial_state,
    std::uint16_t* output,
    float* final_state,
    quactlize_ppu_chunked_gdn_problem_v1 const* problem,
    float scale,
    void* stream) {
  if (problem == nullptr) return QUACTLIZE_PPU_CHUNKED_GDN_NULL_POINTER;
  if (problem->schema_version != QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1) {
    return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  }
  Arguments args{};
  args.q = reinterpret_cast<Element const*>(q);
  args.k = reinterpret_cast<Element const*>(k);
  args.v = reinterpret_cast<Element const*>(v);
  args.gamma_log2_cumsum = gamma_log2_cumsum;
  args.beta = beta;
  args.initial_state = initial_state;
  args.output = reinterpret_cast<Element*>(output);
  args.final_state = final_state;
  args.problem = {
      problem->total_tokens,
      problem->num_sequences,
      problem->sequence_length,
      problem->num_qk_heads,
      problem->num_v_heads,
      problem->head_size_k,
      problem->head_size_v,
      problem->chunk_size,
  };
  args.scale = scale;

  auto const admission = Kernel::Collective::argument_status(args);
  if (admission != cutlass::linear_attention::PpuChunkedGdnStatus::kSuccess) {
    return int(admission);
  }

  int const blocks = Traits::ChunkSize == 64
      ? Kernel::Scheduler::grid_size(args.problem)
      : 0;
  if (blocks <= 0) return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  typename Kernel::SharedStorage* scratch = nullptr;
  if (cudaMalloc(&scratch, std::size_t(blocks) * sizeof(*scratch)) != cudaSuccess) {
    return QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
  }
  cudaStream_t const cuda_stream = static_cast<cudaStream_t>(stream);
  chunked_gdn_global_scratch_kernel<<<blocks, Kernel::MaxThreadsPerBlock, 0, cuda_stream>>>(
      args, scratch);
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) status = cudaStreamSynchronize(cuda_stream);
  cudaError_t const free_status = cudaFree(scratch);
  if (status == cudaSuccess) status = free_status;
  return status == cudaSuccess ? QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS
                               : QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
}

extern "C" std::size_t quactlize_ppu_chunked_gdn_workspace_size_bf16_v2(
    quactlize_ppu_chunked_gdn_problem_v1 const* problem) {
  if (problem == nullptr ||
      problem->schema_version != QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1 ||
      std::int64_t(problem->total_tokens) !=
          std::int64_t(problem->num_sequences) * problem->sequence_length ||
      problem->num_sequences <= 0 || problem->sequence_length <= 0 ||
      problem->num_qk_heads <= 0 || problem->num_v_heads <= 0 ||
      problem->num_v_heads % problem->num_qk_heads != 0 ||
      problem->head_size_k != 128 || problem->head_size_v != 128 ||
      problem->chunk_size != 64) {
    return 0;
  }
  cutlass::linear_attention::PpuChunkedGdnProblem const p{
      problem->total_tokens, problem->num_sequences, problem->sequence_length,
      problem->num_qk_heads, problem->num_v_heads, problem->head_size_k,
      problem->head_size_v, problem->chunk_size};
  return Pipeline::get_workspace_size(p);
}

extern "C" int quactlize_ppu_chunked_gdn_fwd_bf16_v2(
    std::uint16_t const* q,
    std::uint16_t const* k,
    std::uint16_t const* v,
    float const* gamma_log2_cumsum,
    float const* beta,
    float const* initial_state,
    std::uint16_t* output,
    float* final_state,
    quactlize_ppu_chunked_gdn_problem_v1 const* problem,
    float scale,
    void* workspace,
    std::size_t workspace_bytes,
    void* stream) {
  if (problem == nullptr) return QUACTLIZE_PPU_CHUNKED_GDN_NULL_POINTER;
  if (problem->schema_version != QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1) {
    return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  }
  Arguments args = make_arguments(
      q, k, v, gamma_log2_cumsum, beta, initial_state, output, final_state,
      *problem, scale);
  auto const admission = Pipeline::argument_status(args, workspace, workspace_bytes);
  if (admission != cutlass::linear_attention::PpuChunkedGdnStatus::kSuccess) {
    return int(admission);
  }

  auto const params = Pipeline::to_underlying_arguments(args, workspace);
  int const prepare_blocks = Pipeline::PrepareScheduler::grid_size(args.problem);
  int const recurrence_blocks = Pipeline::RecurrenceScheduler::grid_size(args.problem);
  int const scratch_blocks =
      prepare_blocks > recurrence_blocks ? prepare_blocks : recurrence_blocks;
  static_assert(sizeof(typename PrepareKernel::SharedStorage) ==
                    sizeof(typename RecurrenceKernel::SharedStorage),
                "test adapter expects a common scratch ledger");
  typename PrepareKernel::SharedStorage* scratch = nullptr;
  if (scratch_blocks <= 0 ||
      cudaMalloc(&scratch, std::size_t(scratch_blocks) * sizeof(*scratch)) !=
          cudaSuccess) {
    return QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
  }
  cudaStream_t const cuda_stream = static_cast<cudaStream_t>(stream);
  chunked_gdn_prepare_global_scratch_kernel
      <<<prepare_blocks, PrepareKernel::MaxThreadsPerBlock, 0, cuda_stream>>>(
          params, scratch);
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) {
    chunked_gdn_recurrence_global_scratch_kernel
        <<<recurrence_blocks, RecurrenceKernel::MaxThreadsPerBlock, 0,
           cuda_stream>>>(params, scratch);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) status = cudaStreamSynchronize(cuda_stream);
  cudaError_t const free_status = cudaFree(scratch);
  if (status == cudaSuccess) status = free_status;
  return status == cudaSuccess ? QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS
                               : QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
}

extern "C" std::size_t quactlize_ppu_chunked_gdn_workspace_size_bf16_v3(
    quactlize_ppu_chunked_gdn_problem_v1 const* problem) {
  if (problem == nullptr ||
      problem->schema_version != QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1 ||
      std::int64_t(problem->total_tokens) !=
          std::int64_t(problem->num_sequences) * problem->sequence_length ||
      problem->num_sequences <= 0 || problem->sequence_length <= 0 ||
      problem->num_qk_heads <= 0 || problem->num_v_heads <= 0 ||
      problem->num_v_heads % problem->num_qk_heads != 0 ||
      problem->head_size_k != 128 || problem->head_size_v != 128 ||
      problem->chunk_size != 64) {
    return 0;
  }
  cutlass::linear_attention::PpuChunkedGdnProblem const p{
      problem->total_tokens, problem->num_sequences, problem->sequence_length,
      problem->num_qk_heads, problem->num_v_heads, problem->head_size_k,
      problem->head_size_v, problem->chunk_size};
  return FourStagePipeline::get_workspace_size(p);
}

extern "C" int quactlize_ppu_chunked_gdn_fwd_bf16_v3(
    std::uint16_t const* q,
    std::uint16_t const* k,
    std::uint16_t const* v,
    float const* gamma_log2_cumsum,
    float const* beta,
    float const* initial_state,
    std::uint16_t* output,
    float* final_state,
    quactlize_ppu_chunked_gdn_problem_v1 const* problem,
    float scale,
    void* workspace,
    std::size_t workspace_bytes,
    void* stream) {
  if (problem == nullptr) return QUACTLIZE_PPU_CHUNKED_GDN_NULL_POINTER;
  if (problem->schema_version != QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1) {
    return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  }
  Arguments args = make_arguments(
      q, k, v, gamma_log2_cumsum, beta, initial_state, output, final_state,
      *problem, scale);
  auto const admission =
      FourStagePipeline::argument_status(args, workspace, workspace_bytes);
  if (admission != cutlass::linear_attention::PpuChunkedGdnStatus::kSuccess) {
    return int(admission);
  }

  auto const params = FourStagePipeline::to_underlying_arguments(args, workspace);
  int const prepare_blocks =
      FourStagePipeline::PrepareScheduler::grid_size(args.problem);
  int const value_blocks =
      FourStagePipeline::ValueScheduler::grid_size(args.problem);
  int const h_blocks =
      FourStagePipeline::RecurrenceScheduler::grid_size(args.problem);
  int const scratch_blocks =
      prepare_blocks > value_blocks
          ? (prepare_blocks > h_blocks ? prepare_blocks : h_blocks)
          : (value_blocks > h_blocks ? value_blocks : h_blocks);
  static_assert(sizeof(typename FourPrepareKernel::SharedStorage) == 57856 &&
                    sizeof(typename UKernel::SharedStorage) == 41472 &&
                    sizeof(typename HKernel::SharedStorage) == 98816 &&
                    sizeof(typename OKernel::SharedStorage) == 66048 &&
                    kFourStageScratchStride >=
                        sizeof(typename FourPrepareKernel::SharedStorage) &&
                    kFourStageScratchStride >=
                        sizeof(typename UKernel::SharedStorage) &&
                    kFourStageScratchStride >=
                        sizeof(typename OKernel::SharedStorage),
                "test adapter must cover every stage-specific scratch ledger");
  std::uint8_t* scratch = nullptr;
  if (scratch_blocks <= 0 ||
      cudaMalloc(&scratch,
                 std::size_t(scratch_blocks) * kFourStageScratchStride) !=
          cudaSuccess) {
    return QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
  }
  cudaStream_t const cuda_stream = static_cast<cudaStream_t>(stream);
  chunked_gdn_four_stage_global_scratch_kernel<FourPrepareKernel>
      <<<prepare_blocks, FourPrepareKernel::MaxThreadsPerBlock, 0, cuda_stream>>>(
          params, scratch);
  cudaError_t status = cudaGetLastError();
  if (status == cudaSuccess) {
    chunked_gdn_four_stage_global_scratch_kernel<UKernel>
        <<<value_blocks, UKernel::MaxThreadsPerBlock, 0, cuda_stream>>>(
            params, scratch);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    chunked_gdn_four_stage_global_scratch_kernel<HKernel>
        <<<h_blocks, HKernel::MaxThreadsPerBlock, 0, cuda_stream>>>(
            params, scratch);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) {
    chunked_gdn_four_stage_global_scratch_kernel<OKernel>
        <<<value_blocks, OKernel::MaxThreadsPerBlock, 0, cuda_stream>>>(
            params, scratch);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess) status = cudaStreamSynchronize(cuda_stream);
  cudaError_t const free_status = cudaFree(scratch);
  if (status == cudaSuccess) status = free_status;
  return status == cudaSuccess ? QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS
                               : QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
}
