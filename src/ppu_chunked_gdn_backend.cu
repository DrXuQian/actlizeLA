// Device-library entry for the pure C++ PPU chunked-GDN forward kernel.

#include <cstdint>

#include "cutlass/bfloat16.h"
#include "cutlass/device_kernel.h"
#include "quactlize_ppu_linear_attention.h"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_kernel.cuh"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_pipeline.cuh"

#if defined(__HGGCCC__)
#include <hggc_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace {

using Element = cutlass::bfloat16_t;
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
using Arguments = cutlass::linear_attention::PpuChunkedGdnArguments<Element, Element, float>;
using Kernel = cutlass::linear_attention::PpuChunkedGdnKernel<Arguments, Traits>;
using Pipeline =
    cutlass::linear_attention::PpuChunkedGdnTwoStagePipeline<Arguments, Traits>;
using PrepareKernel = cutlass::linear_attention::PpuChunkedGdnPrepareKernel<Pipeline>;
using RecurrenceKernel =
    cutlass::linear_attention::PpuChunkedGdnRecurrenceKernel<Pipeline>;
using Admission = cutlass::linear_attention::PpuChunkedGdnStatus;

static_assert(int(Admission::kSuccess) == QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS &&
                  int(Admission::kNullPointer) == QUACTLIZE_PPU_CHUNKED_GDN_NULL_POINTER &&
                  int(Admission::kInvalidProblem) == QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM &&
                  int(Admission::kUnsupportedHeadDimension) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_HEAD_DIMENSION &&
                  int(Admission::kUnsupportedChunkSize) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_CHUNK_SIZE &&
                  int(Admission::kUnsupportedHeadMapping) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_UNSUPPORTED_HEAD_MAPPING &&
                  int(Admission::kInvalidSequenceLayout) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_INVALID_SEQUENCE_LAYOUT &&
                  int(Admission::kMisalignedPointer) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_MISALIGNED_POINTER &&
                  int(Admission::kInsufficientWorkspace) ==
                      QUACTLIZE_PPU_CHUNKED_GDN_INSUFFICIENT_WORKSPACE,
              "public C status values must match the CUTLASS admission ABI");

template <class DeviceKernel>
int configure_dynamic_shared_memory() {
  constexpr int bytes = int(sizeof(typename DeviceKernel::SharedStorage));
  if (bytes < (48 << 10)) return QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS;
#if defined(__HGGCCC__)
  hggcError_t const result = hggcFuncSetAttribute(
      cutlass::device_kernel<DeviceKernel>,
      hggcFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (result != hggcSuccess) {
    (void)hggcGetLastError();
    return QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
  }
#else
  cudaError_t const result = cudaFuncSetAttribute(
      cutlass::device_kernel<DeviceKernel>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (result != cudaSuccess) {
    (void)cudaGetLastError();
    return QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
  }
#endif
  return QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS;
}

void clear_runtime_error() {
#if defined(__HGGCCC__)
  (void)hggcGetLastError();
#else
  (void)cudaGetLastError();
#endif
}

bool launch_succeeded() {
#if defined(__HGGCCC__)
  return hggcPeekAtLastError() == hggcSuccess;
#else
  return cudaPeekAtLastError() == cudaSuccess;
#endif
}

bool valid_schema(quactlize_ppu_chunked_gdn_problem_v1 const* problem) {
  return problem != nullptr &&
         problem->schema_version == QUACTLIZE_PPU_CHUNKED_GDN_SCHEMA_V1;
}

cutlass::linear_attention::PpuChunkedGdnProblem convert_problem(
    quactlize_ppu_chunked_gdn_problem_v1 const& problem) {
  return {
      problem.total_tokens,
      problem.num_sequences,
      problem.sequence_length,
      problem.num_qk_heads,
      problem.num_v_heads,
      problem.head_size_k,
      problem.head_size_v,
      problem.chunk_size,
  };
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
  args.problem = convert_problem(problem);
  args.scale = scale;
  return args;
}

}  // namespace

extern "C" int quactlize_ppu_chunked_gdn_fwd_bf16_v1(
    uint16_t const* q,
    uint16_t const* k,
    uint16_t const* v,
    float const* gamma_log2_cumsum,
    float const* beta,
    float const* initial_state,
    uint16_t* output,
    float* final_state,
    quactlize_ppu_chunked_gdn_problem_v1 const* problem,
    float scale,
    void* stream) {
  if (problem == nullptr) return QUACTLIZE_PPU_CHUNKED_GDN_NULL_POINTER;
  if (!valid_schema(problem)) {
    return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  }

  Arguments args = make_arguments(
      q, k, v, gamma_log2_cumsum, beta, initial_state, output, final_state,
      *problem, scale);

  auto const status = Kernel::Collective::argument_status(args);
  if (status != cutlass::linear_attention::PpuChunkedGdnStatus::kSuccess) {
    return int(status);
  }
  if (!Kernel::can_implement(args)) {
    return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  }
  int const smem_status = configure_dynamic_shared_memory<Kernel>();
  if (smem_status != QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS) return smem_status;

  clear_runtime_error();
  auto const params = Kernel::to_underlying_arguments(args, nullptr);
#if defined(__HGGCCC__)
  hggcStream_t const launch_stream = static_cast<hggcStream_t>(stream);
#else
  cudaStream_t const launch_stream = static_cast<cudaStream_t>(stream);
#endif
  cutlass::device_kernel<Kernel>
      <<<Kernel::get_grid_shape(params), Kernel::get_block_shape(),
         sizeof(typename Kernel::SharedStorage), launch_stream>>>(params);
  return launch_succeeded()
      ? QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS
      : QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
}

extern "C" std::size_t quactlize_ppu_chunked_gdn_workspace_size_bf16_v2(
    quactlize_ppu_chunked_gdn_problem_v1 const* problem) {
  if (!valid_schema(problem)) return 0;
  auto const p = convert_problem(*problem);
  std::int64_t const tokens =
      std::int64_t(p.num_sequences) * std::int64_t(p.sequence_length);
  if (p.total_tokens <= 0 || tokens != p.total_tokens ||
      p.num_sequences <= 0 || p.sequence_length <= 0 ||
      p.num_qk_heads <= 0 || p.num_v_heads <= 0 ||
      p.num_v_heads % p.num_qk_heads != 0 ||
      p.head_size_k != Traits::HeadSizeK ||
      p.head_size_v != Traits::HeadSizeV || p.chunk_size != Traits::ChunkSize) {
    return 0;
  }
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
  if (!valid_schema(problem)) {
    return QUACTLIZE_PPU_CHUNKED_GDN_INVALID_PROBLEM;
  }
  Arguments args = make_arguments(
      q, k, v, gamma_log2_cumsum, beta, initial_state, output, final_state,
      *problem, scale);
  Admission const status =
      Pipeline::argument_status(args, workspace, workspace_bytes);
  if (status != Admission::kSuccess) return int(status);

  int status_prepare = configure_dynamic_shared_memory<PrepareKernel>();
  if (status_prepare != QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS) return status_prepare;
  int status_recurrence = configure_dynamic_shared_memory<RecurrenceKernel>();
  if (status_recurrence != QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS) {
    return status_recurrence;
  }

  clear_runtime_error();
  auto const params = Pipeline::to_underlying_arguments(args, workspace);
#if defined(__HGGCCC__)
  hggcStream_t const launch_stream = static_cast<hggcStream_t>(stream);
#else
  cudaStream_t const launch_stream = static_cast<cudaStream_t>(stream);
#endif
  cutlass::device_kernel<PrepareKernel>
      <<<PrepareKernel::get_grid_shape(params), PrepareKernel::get_block_shape(),
         sizeof(typename PrepareKernel::SharedStorage), launch_stream>>>(params);
  if (!launch_succeeded()) return QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
  cutlass::device_kernel<RecurrenceKernel>
      <<<RecurrenceKernel::get_grid_shape(params),
         RecurrenceKernel::get_block_shape(),
         sizeof(typename RecurrenceKernel::SharedStorage), launch_stream>>>(params);
  return launch_succeeded()
      ? QUACTLIZE_PPU_CHUNKED_GDN_SUCCESS
      : QUACTLIZE_PPU_CHUNKED_GDN_RUNTIME_ERROR;
}
