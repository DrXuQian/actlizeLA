// Full-C64 bounds only; causal masking and all HV output arithmetic retained.
#include "gdn_wy_common.cuh"
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_residual_warps8_hvlayout.cuh"

namespace gdn_qsa::wy::full_chunk_output {
using namespace residual_warps8_hvlayout;
__global__ void __launch_bounds__(OutputTile::Threads)
gdn_wy_full_chunk_output(Inputs p, Workspace ws, BF16* output) {
  using Q = aiu::Tile<Chunk, Dim>;
  using A = aiu::Tile<Chunk, Chunk>;
  using H = OutputSnapshot;
  using V = OutputValue;
  using Output = aiu::Tile<Chunk, OutputTile::Panel>;  // public output stays row-oriented
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<TiledOutputStorage*>(storage);
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks(), bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads, qh = p.shape.q_head(h);
  int const first = ct * Chunk, valid = Chunk;
  int64_t const group = p.shape.group(b, h, ct);
  Q::stage(sm.q, p.q + p.shape.input(b, first, qh, p.shape.q_heads), p.shape.q_heads * Dim, valid);
  Q::stage(sm.stage.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads), p.shape.q_heads * Dim, valid);
  if (tid < Chunk) sm.g[tid] = ws.gates[group * Chunk + tid];
  commit_wait();
  float attention[OutputTile::Fragments][8] = {};
  CUTE_UNROLL
  for (int k = 0; k < Dim; k += 16) {
    uint32_t q[4];
    Q::load(sm.q, OutputTile::row(warp), k, q);
    CUTE_UNROLL
    for (int c = 0; c < OutputTile::Fragments; ++c) {
      uint32_t key[4];
      Q::load(sm.stage.k, OutputTile::column(warp, c), k, key);
      bf16_mma(attention[c], q, key);
    }
  }
  CUTE_UNROLL
  for (int c = 0; c < OutputTile::Fragments; ++c) {
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = OutputTile::row(warp) + rc.row, col = OutputTile::column(warp, c) + rc.col;
      sm.attn[A::offset(row, col)] = BF16(
          row >= col ? attention[c][s] * expf(sm.g[row] - sm.g[col]) : 0.0f);
    }
  }
  __syncthreads();
  float row_gate[2];
  CUTE_UNROLL
  for (int row = 0; row < 2; ++row)
    row_gate[row] = expf(sm.g[OutputTile::row(warp) + lane / 4 + row * 8]);
  #pragma unroll 1
  for (int panel = 0; panel < Dim; panel += OutputTile::Panel) {
    H::stage(sm.stage.h, ws.snapshots + Snapshot::workspace_offset(group, 0, panel));
    V::stage(sm.v, ws.vnew + PublishedValue::workspace_offset(group, 0, panel));
    commit_wait();
    float acc[OutputTile::Fragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t q[4];
      Q::load(sm.q, OutputTile::row(warp), k, q);
      CUTE_UNROLL
      for (int c = 0; c < OutputTile::Fragments; ++c) {
        uint32_t hreg[4];
        H::load(sm.stage.h, k, OutputTile::column(warp, c), hreg);
        bf16_mma(acc[c], q, hreg);
      }
    }
    CUTE_UNROLL
    for (int c = 0; c < OutputTile::Fragments; ++c) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) acc[c][s] *= row_gate[s / 4];
    }
    CUTE_UNROLL
    for (int k = 0; k < Chunk; k += 16) {
      uint32_t a[4];
      A::load(sm.attn, OutputTile::row(warp), k, a);
      CUTE_UNROLL
      for (int c = 0; c < OutputTile::Fragments; ++c) {
        uint32_t v[4];
        V::load(sm.v, k, OutputTile::column(warp, c), v);
        bf16_mma(acc[c], a, v);
      }
    }
    __syncthreads();  // the union's H readers finish before output exchange
    CUTE_UNROLL
    for (int c = 0; c < OutputTile::Fragments; ++c) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.stage.output[Output::offset(OutputTile::row(warp) + rc.row,
                                  OutputTile::column(warp, c) + rc.col)] =
            BF16(acc[c][s] * 0.08838834764831845f);
      }
    }
    __syncthreads();
    Output::publish<OutputTile::Threads>(sm.stage.output,
        output + p.shape.input(b, first, h, p.shape.value_heads) + panel,
        int64_t(p.shape.value_heads) * Dim, valid);
    __syncthreads();
  }
}

int configure() {
  return int(hggcFuncSetAttribute(gdn_wy_full_chunk_output,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledOutputStorage)));
}
int launch_output(Inputs p, Workspace ws, BF16* output, gdn_arch::Stream stream) {
  gdn_wy_full_chunk_output<<<unsigned(p.shape.groups()), OutputTile::Threads,
                            sizeof(TiledOutputStorage), stream>>>(p, ws, output);
  return int(hggcGetLastError());
}
}  // namespace gdn_qsa::wy::full_chunk_output
