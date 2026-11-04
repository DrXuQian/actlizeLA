#include "common.cuh"

namespace actlize_la::ppu15 {
namespace {
struct OutputStorage {
  alignas(128) BF16 q[Chunk * Dim], attention[Chunk * Chunk];
  union {
    alignas(128) BF16 k[Chunk * Dim];
    alignas(128) BF16 h[OutputTile::Panel * Dim];
    alignas(128) BF16 output[Chunk * OutputTile::Panel];
  } stage;
  alignas(128) BF16 value[OutputTile::Panel * Chunk];
  float prefix[Chunk];
};
static_assert(sizeof(OutputStorage) == 49408);
using Q = Tile<Chunk, Dim>;
using Attention = Tile<Chunk, Chunk>;
using H = Tile<OutputTile::Panel, Dim>;
using V = Tile<OutputTile::Panel, Chunk>;
using Output = Tile<Chunk, OutputTile::Panel>;

__global__ void __launch_bounds__(OutputTile::Threads) output_kernel(Inputs p, Workspace ws, BF16* output) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<OutputStorage*>(storage);
  int const tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks(), bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads, qh = p.shape.q_head(h);
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int64_t const group = p.shape.group(b, h, ct);
  Q::stage(sm.q, p.q + p.shape.input(b, first, qh, p.shape.q_heads), p.shape.q_heads * Dim, valid);
  Q::stage(sm.stage.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads), p.shape.q_heads * Dim, valid);
  if (tid < Chunk) sm.prefix[tid] = ws.gates[group * Chunk + tid];
  commit_wait();
  float attention[OutputTile::Fragments][8] = {};
  CUTE_UNROLL
  for (int k = 0; k < Dim; k += 16) {
    uint32_t a[4];
    Q::load<true>(sm.q, OutputTile::row(warp), k, a);
    CUTE_UNROLL
    for (int f = 0; f < OutputTile::Fragments; ++f) {
      uint32_t bs[4];
      Q::load<false>(sm.stage.k, OutputTile::col(warp, f), k, bs);
      bf16_mma(attention[f], a, bs);
    }
  }
  CUTE_UNROLL
  for (int f = 0; f < OutputTile::Fragments; ++f) {
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = OutputTile::row(warp) + rc.row, col = OutputTile::col(warp, f) + rc.col;
      sm.attention[Attention::offset(row, col)] = BF16(row >= col && row < valid
          ? attention[f][s] * expf(sm.prefix[row] - sm.prefix[col]) : 0.0f);
    }
  }
  __syncthreads();
  float row_gate[2];
  CUTE_UNROLL
  for (int r = 0; r < 2; ++r) row_gate[r] = expf(sm.prefix[OutputTile::row(warp) + lane / 4 + r * 8]);
  #pragma unroll 1
  for (int panel = 0; panel < Dim; panel += OutputTile::Panel) {
    H::stage(sm.stage.h, ws.snapshots + snapshot_base(group) + panel * Dim, Dim, OutputTile::Panel);
    V::stage(sm.value, ws.vnew + value_base(group) + panel * Chunk, Chunk, OutputTile::Panel);
    commit_wait();
    float acc[OutputTile::Fragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t a[4];
      Q::load<true>(sm.q, OutputTile::row(warp), k, a);
      CUTE_UNROLL
      for (int f = 0; f < OutputTile::Fragments; ++f) {
        uint32_t bs[4];
        H::load<false>(sm.stage.h, OutputTile::col(warp, f), k, bs);
        bf16_mma(acc[f], a, bs);
      }
    }
    CUTE_UNROLL
    for (int f = 0; f < OutputTile::Fragments; ++f) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) acc[f][s] *= row_gate[row_half(s)];
    }
    CUTE_UNROLL
    for (int k = 0; k < Chunk; k += 16) {
      uint32_t a[4];
      Attention::load<true>(sm.attention, OutputTile::row(warp), k, a);
      CUTE_UNROLL
      for (int f = 0; f < OutputTile::Fragments; ++f) {
        uint32_t bs[4];
        V::load<false>(sm.value, OutputTile::col(warp, f), k, bs);
        bf16_mma(acc[f], a, bs);
      }
    }
    __syncthreads();  // H readers retire before the union becomes output
    CUTE_UNROLL
    for (int f = 0; f < OutputTile::Fragments; ++f) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.stage.output[Output::offset(OutputTile::row(warp) + rc.row, OutputTile::col(warp, f) + rc.col)] =
            BF16(acc[f][s] * 0.08838834764831845f);  // 1/sqrt(128), as retained path
      }
    }
    __syncthreads();
    Output::publish<OutputTile::Threads>(sm.stage.output,
        output + p.shape.input(b, first, h, p.shape.value_heads) + panel, int64_t(p.shape.value_heads) * Dim, valid);
    __syncthreads();
  }
}
}  // namespace
int configure_output() {
  return int(hggcFuncSetAttribute(output_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(OutputStorage)));
}
int output(Inputs p, Workspace ws, BF16* result, hggcStream_t stream) {
  output_kernel<<<unsigned(p.shape.groups()), OutputTile::Threads, sizeof(OutputStorage), stream>>>(p, ws, result);
  return int(hggcGetLastError());
}
}  // namespace actlize_la::ppu15
