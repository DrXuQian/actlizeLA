// L204: instantiate the complete chunked-GDN device body with the shipping
// BF16/C64/K128/V128 type.  It launches nothing; L203 owns algebra and the PPU
// box owns hardware opcodes.  This gate proves the CUDA/CUTLASS body and public
// launch geometry are one compilable type rather than disconnected headers.

#include <cstdio>
#include <cstdint>

#include "cutlass/bfloat16.h"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_kernel.cuh"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_pipeline.cuh"
#include "actlize_extensions/cutlass/linear_attention/ppu_chunked_gdn_triton_pipeline.cuh"

namespace {

#if defined(L204_PLANT_CHUNK)
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<32, 128, 128>;
#elif defined(L204_PLANT_HEAD)
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 64, 128>;
#else
using Traits = cutlass::linear_attention::PpuChunkedGdnTraits<64, 128, 128>;
#endif
using Args = cutlass::linear_attention::PpuChunkedGdnArguments<
    cutlass::bfloat16_t, cutlass::bfloat16_t, float>;
using Kernel = cutlass::linear_attention::PpuChunkedGdnKernel<Args, Traits>;
using Pipeline = cutlass::linear_attention::PpuChunkedGdnTwoStagePipeline<Args, Traits>;
using PrepareKernel = cutlass::linear_attention::PpuChunkedGdnPrepareKernel<Pipeline>;
using RecurrenceKernel =
    cutlass::linear_attention::PpuChunkedGdnRecurrenceKernel<Pipeline>;
using FourStagePipeline =
    cutlass::linear_attention::PpuChunkedGdnFourStagePipeline<Args, Traits>;
using FourPrepareKernel =
    cutlass::linear_attention::PpuChunkedGdnFourStagePrepareKernel<FourStagePipeline>;
using UKernel = cutlass::linear_attention::PpuChunkedGdnUKernel<FourStagePipeline>;
using HKernel = cutlass::linear_attention::PpuChunkedGdnHKernel<FourStagePipeline>;
using OKernel = cutlass::linear_attention::PpuChunkedGdnOKernel<FourStagePipeline>;
using TritonPipeline =
    cutlass::linear_attention::PpuChunkedGdnTritonPipeline<Args, Traits>;
using TritonKktKernel =
    cutlass::linear_attention::PpuChunkedGdnTritonKktKernel<TritonPipeline>;
using TritonWuKernel =
    cutlass::linear_attention::PpuChunkedGdnTritonWuKernel<TritonPipeline>;
using TritonHKernel =
    cutlass::linear_attention::PpuChunkedGdnTritonHKernel<TritonPipeline>;
using TritonOKernel =
    cutlass::linear_attention::PpuChunkedGdnTritonOKernel<TritonPipeline>;

static_assert(Kernel::Collective::kAllStagesConnected,
              "L204 requires the complete GDN dataflow");
static_assert(Kernel::Collective::kGlobalQkAndKkUseAiuOnPpu0010,
              "L204 requires the proved PPU global-dot AIU route");
static_assert(Kernel::Collective::kGeneratedOperandMmaConnected &&
                  Kernel::Collective::kAllDenseForwardProductsUseAiu,
              "L204 requires every dense forward GEMM product on PPU AIU");
static_assert(Kernel::Collective::kInverseBlockUpdatesUseAiu &&
                  Kernel::Collective::kAllMatrixProductsUseAiu,
              "L204 requires inverse block updates on PPU TF32 AIU");
static_assert(Kernel::Scheduler::ValueTiles == 2 &&
                  Kernel::Collective::kGeneratedProductKinds == 6 &&
                  Kernel::Collective::kGeneratedProductInstancesPerWorkTileChunk == 6 &&
                  Kernel::Collective::kGeneratedMmaPerWorkTileChunk == 640 &&
                  Kernel::Collective::kGlobalDotMmaPerWorkTileChunk == 256 &&
                  Kernel::Collective::kInverseBlockProductsPerChunk == 6 &&
                  Kernel::Collective::kInverseTf32MmaPerWorkTileChunk == 40 &&
                  Kernel::Collective::kInverseCtaBarriersPerChunk == 8 &&
                  Kernel::Collective::kBf16MmaPerWorkTileChunk == 896 &&
                  Kernel::Collective::kDenseForwardMmaPerWorkTileChunk == 936 &&
                  Kernel::Collective::kBf16MmaPerLogicalHeadChunk == 1792 &&
                  Kernel::Collective::kTf32MmaPerLogicalHeadChunk == 80,
              "L204 split-V BF16/TF32 execution denominators changed");
static_assert(Kernel::Collective::kPreparedChunkBytes == 32768 &&
                  Kernel::Collective::kPreparedValueChunkBytes == 40960 &&
                  Kernel::Collective::kPrepareBf16MmaPerChunk == 384 &&
                  Kernel::Collective::kRecurrenceBf16MmaPerValueTileChunk == 512 &&
                  Kernel::Collective::kTwoStageBf16MmaPerLogicalHeadChunk == 1408 &&
                  Kernel::Collective::kTwoStageTf32MmaPerLogicalHeadChunk == 40,
              "L204 two-stage workspace or execution denominator changed");
static_assert(Kernel::MaxThreadsPerBlock == 128,
              "L204 launch geometry changed without a new proof");
static_assert(sizeof(typename Kernel::SharedStorage) <= 262144,
              "L204 exceeds the PPU per-CTA shared-memory budget");
static_assert(sizeof(typename FourPrepareKernel::SharedStorage) == 57856 &&
                  sizeof(typename UKernel::SharedStorage) == 41472 &&
                  sizeof(typename HKernel::SharedStorage) == 98816 &&
                  sizeof(typename OKernel::SharedStorage) == 66048,
              "L204 four-stage storage must contain only each stage's live set");
static_assert(TritonPipeline::Collective::kChunkWorkspaceBytes == 90112 &&
                  sizeof(typename TritonKktKernel::SharedStorage) == 33280 &&
                  sizeof(typename TritonWuKernel::SharedStorage) == 8704 &&
                  sizeof(typename TritonHKernel::SharedStorage) == 41472 &&
                  sizeof(typename TritonOKernel::SharedStorage) == 8704,
              "L204 Triton-aligned seams changed");

// actlize's generic device_kernel ends in an hgcc-only synclog call, so plain
// nvcc cannot use that wrapper as a portability gate.  This minimal equivalent
// still instantiates the shipping operator and every reachable scalar phase;
// the hgcc backend separately launches the same operator through device_kernel.
__global__ void nvcc_device_kernel(typename Kernel::Params params) {
  extern __shared__ char smem[];
  Kernel{}(params, smem);
}

__global__ void nvcc_prepare_kernel(typename Pipeline::Params params) {
  extern __shared__ char smem[];
  PrepareKernel{}(params, smem);
}

__global__ void nvcc_recurrence_kernel(typename Pipeline::Params params) {
  extern __shared__ char smem[];
  RecurrenceKernel{}(params, smem);
}

template <class DeviceKernel>
__global__ void nvcc_four_stage_kernel(typename FourStagePipeline::Params params) {
  extern __shared__ char smem[];
  DeviceKernel{}(params, smem);
}

template <class DeviceKernel>
__global__ void nvcc_triton_stage_kernel(typename TritonPipeline::Params params) {
  extern __shared__ char smem[];
  DeviceKernel{}(params, smem);
}

// The launch expression forces nvcc to instantiate the complete operator.
// main() deliberately does not call it.
void instantiate_device_body(typename Kernel::Params params) {
  nvcc_device_kernel
      <<<Kernel::get_grid_shape(params), Kernel::get_block_shape(),
         sizeof(typename Kernel::SharedStorage)>>>(params);
}

void instantiate_two_stage_body(typename Pipeline::Params params) {
  nvcc_prepare_kernel
      <<<PrepareKernel::get_grid_shape(params), PrepareKernel::get_block_shape(),
         sizeof(typename PrepareKernel::SharedStorage)>>>(params);
  nvcc_recurrence_kernel
      <<<RecurrenceKernel::get_grid_shape(params),
         RecurrenceKernel::get_block_shape(),
         sizeof(typename RecurrenceKernel::SharedStorage)>>>(params);
}

void instantiate_four_stage_body(typename FourStagePipeline::Params params) {
  nvcc_four_stage_kernel<FourPrepareKernel>
      <<<FourPrepareKernel::get_grid_shape(params),
         FourPrepareKernel::get_block_shape(),
         sizeof(typename FourPrepareKernel::SharedStorage)>>>(params);
  nvcc_four_stage_kernel<UKernel>
      <<<UKernel::get_grid_shape(params), UKernel::get_block_shape(),
         sizeof(typename UKernel::SharedStorage)>>>(params);
  nvcc_four_stage_kernel<HKernel>
      <<<HKernel::get_grid_shape(params), HKernel::get_block_shape(),
         sizeof(typename HKernel::SharedStorage)>>>(params);
  nvcc_four_stage_kernel<OKernel>
      <<<OKernel::get_grid_shape(params), OKernel::get_block_shape(),
         sizeof(typename OKernel::SharedStorage)>>>(params);
}

void instantiate_triton_aligned_body(typename TritonPipeline::Params params) {
  nvcc_triton_stage_kernel<TritonKktKernel>
      <<<TritonKktKernel::get_grid_shape(params),
         TritonKktKernel::get_block_shape(),
         sizeof(typename TritonKktKernel::SharedStorage)>>>(params);
  nvcc_triton_stage_kernel<TritonWuKernel>
      <<<TritonWuKernel::get_grid_shape(params),
         TritonWuKernel::get_block_shape(),
         sizeof(typename TritonWuKernel::SharedStorage)>>>(params);
  nvcc_triton_stage_kernel<TritonHKernel>
      <<<TritonHKernel::get_grid_shape(params),
         TritonHKernel::get_block_shape(),
         sizeof(typename TritonHKernel::SharedStorage)>>>(params);
  nvcc_triton_stage_kernel<TritonOKernel>
      <<<TritonOKernel::get_grid_shape(params),
         TritonOKernel::get_block_shape(),
         sizeof(typename TritonOKernel::SharedStorage)>>>(params);
}

}  // namespace

int main() {
  alignas(16) std::uint16_t qkv[8]{};
  float scalar = 0.0f;
  Args args{};
  args.q = args.k = args.v =
      reinterpret_cast<cutlass::bfloat16_t const*>(qkv);
  args.gamma_log2_cumsum = args.beta = &scalar;
  args.output = reinterpret_cast<cutlass::bfloat16_t*>(qkv);
  args.problem = {65, 1, 65, 1, 1, 128, 128, 64};
  args.scale = 0.5f;

  bool const admitted = Kernel::can_implement(args);
  auto const params = Kernel::to_underlying_arguments(args, nullptr);
  std::size_t const workspace_bytes = Pipeline::get_workspace_size(args.problem);
  auto const pipeline_params = Pipeline::to_underlying_arguments(args, qkv);
  dim3 const grid = Kernel::get_grid_shape(params);
  dim3 const block = Kernel::get_block_shape();
  dim3 const prepare_grid = PrepareKernel::get_grid_shape(pipeline_params);
  dim3 const recurrence_grid = RecurrenceKernel::get_grid_shape(pipeline_params);
  std::size_t const four_workspace_bytes =
      FourStagePipeline::get_workspace_size(args.problem);
  auto const four_params = FourStagePipeline::to_underlying_arguments(args, qkv);
  dim3 const four_prepare_grid = FourPrepareKernel::get_grid_shape(four_params);
  dim3 const u_grid = UKernel::get_grid_shape(four_params);
  dim3 const h_grid = HKernel::get_grid_shape(four_params);
  dim3 const o_grid = OKernel::get_grid_shape(four_params);
  std::size_t const triton_workspace_bytes =
      TritonPipeline::get_workspace_size(args.problem);
  auto const triton_params =
      TritonPipeline::to_underlying_arguments(args, qkv);
  dim3 const triton_kkt_grid = TritonKktKernel::get_grid_shape(triton_params);
  dim3 const triton_wu_grid = TritonWuKernel::get_grid_shape(triton_params);
  dim3 const triton_h_grid = TritonHKernel::get_grid_shape(triton_params);
  dim3 const triton_o_grid = TritonOKernel::get_grid_shape(triton_params);
  bool const ok = admitted && grid.x == 2 && grid.y == 1 && grid.z == 1 &&
                  block.x == 128 && Kernel::get_workspace_size(args) == 0 &&
                  workspace_bytes == 2u * 32768u && prepare_grid.x == 2 &&
                  recurrence_grid.x == 2 &&
                  four_workspace_bytes == 2u * 32768u + 4u * 40960u &&
                  four_prepare_grid.x == 2 && u_grid.x == 4 &&
                  h_grid.x == 2 && o_grid.x == 4 &&
                  triton_workspace_bytes == 2u * 90112u &&
                  triton_kkt_grid.x == 2 && triton_wu_grid.x == 2 &&
                  triton_h_grid.x == 2 && triton_o_grid.x == 4;
  std::printf(
      "[l204] %s: device-body=INSTANTIATED C=64 K=128 V=128 threads=%u "
      "shared=%zu value-tiles=2 all-stages=1 global-dot=PPU-AIU "
      "generated-products/work-tile=6/640 bf16-mma/work-tile=896 "
      "bf16-mma/logical-head=1792 inverse-block-products/work-tile=6 "
      "inverse-tf32-mma/work-tile=40 inverse-tf32-mma/logical-head=80 "
      "inverse-cta-barriers/work-tile=8 "
      "all-matrix-products=AIU inverse-base=16x16-sequential "
      "two-stage=A+W+P/32768B prepare-grid=%u recurrence-grid=%u "
      "bf16-mma/logical-head=1408 tf32-mma/logical-head=40 "
      "four-stage-grid=%u/%u/%u/%u seams=32768B+40960B "
      "four-stage-shared=57856/41472/98816/66048B "
      "triton-aligned-grid=%u/%u/%u/%u seam=90112B "
      "shared=33280/8704/41472/8704B\n",
      ok ? "PASS" : "FAIL", unsigned(block.x),
      sizeof(typename Kernel::SharedStorage), unsigned(prepare_grid.x),
      unsigned(recurrence_grid.x), unsigned(four_prepare_grid.x),
      unsigned(u_grid.x), unsigned(h_grid.x), unsigned(o_grid.x),
      unsigned(triton_kkt_grid.x), unsigned(triton_wu_grid.x),
      unsigned(triton_h_grid.x), unsigned(triton_o_grid.x));
  return ok ? 0 : 1;
}
