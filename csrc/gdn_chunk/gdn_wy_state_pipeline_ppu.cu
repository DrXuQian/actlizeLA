// Single-buffer operand pipeline. Only scheduling changes from the admitted
// AIU state kernel: preserve every accumulator, rounding boundary and K order.
#include "gdn_wy_common.cuh"
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_aiu.cuh"
#include "gdn_qsa/ppu/wy_state_pipeline.cuh"

namespace gdn_qsa::wy::state_pipeline {

__global__ void __launch_bounds__(Plan::Threads)
gdn_wy_state_pipeline(Inputs p, Workspace ws, float* final) {
  using W = aiu::Tile<Chunk, Dim>;
  using H = aiu::Tile<Dim, ValueTile>;
  using V = aiu::Tile<Chunk, ValueTile>;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<TiledStateStorage*>(storage);
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const slice = int(blockIdx.x) % (Dim / ValueTile);
  int const bh = int(blockIdx.x) / (Dim / ValueTile);
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const v0 = slice * ValueTile, qh = p.shape.q_head(h);
  float state[StateTile::KFragments][8];
  CUTE_UNROLL
  for (int k = 0; k < StateTile::KFragments; ++k) {
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = StateTile::k_row(warp, k) + rc.row;
      int const col = v0 + StateTile::column(warp) + rc.col;
      state[k][s] = p.initial ? p.initial[(int64_t(bh) * Dim + row) * Dim + col] : 0.0f;
    }
  }
  // PROLOGUE: only the first W, committed before entering the recurrence.
  W::stage(sm.w, ws.w + tile_offset(p.shape.group(b, h, 0)), Dim, Chunk);
  cute::cp_async_fence();
  #pragma unroll 1
  for (int ct = 0; ct < p.shape.chunks(); ++ct) {
    int64_t const group = p.shape.group(b, h, ct);
    int const first = ct * Chunk, valid = Plan::valid(ct, p.shape.sequence);
    // SNAPSHOT: these stores overlap the already-issued W load. Snapshot
    // readers of the previous iteration retired at INPUTS_READY below.
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.snapshot[H::offset(StateTile::k_row(warp, k) + rc.row,
                               StateTile::column(warp) + rc.col)] = BF16(state[k][s]);
      }
    }
    // W_READY also retires ALL previous K/U/scaledV readers before reuse.
    cute::cp_async_wait<0>();
    __syncthreads();
    H::publish<StateTile::Threads>(sm.snapshot, ws.snapshots + state_offset(group) + v0, Dim);
    // CURRENT_INPUTS: K is not needed by W@H, so do not wait for it up front.
    W::stage(sm.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
             p.shape.q_heads * Dim, valid);
    V::stage(sm.u, ws.u + tile_offset(group) + v0, Dim, Chunk);
    if (tid < Chunk) sm.g[tid] = ws.gates[group * Chunk + tid];
    cute::cp_async_fence();
    // PROJECT: exact admitted W@H arithmetic, concurrent with K/U delivery.
    float value[StateTile::ValueFragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t hs[4];
      H::load<true>(sm.snapshot, k, StateTile::column(warp), hs);
      CUTE_UNROLL
      for (int r = 0; r < StateTile::ValueFragments; ++r) {
        uint32_t w[4];
        W::load(sm.w, StateTile::value_row(warp, r), k, w);
        bf16_mma(value[r], w, hs);
      }
    }
    // INPUTS_READY: retire all W/snapshot readers before W is overwritten.
    cute::cp_async_wait<0>();
    __syncthreads();
    // NEXT_W: no second buffer. Its wait is at NEXT iteration's W_READY,
    // never between this prefetch and the current K^T scaledV computation.
    int const next = Plan::next(ct, p.shape.chunks());
    if (next >= 0) {
      W::stage(sm.w, ws.w + tile_offset(p.shape.group(b, h, next)), Dim, Chunk);
      cute::cp_async_fence();
    }
    // CONDITION: preserve x in FP32 for scaledV, not BF16(vnew)*gate.
    float const last = sm.g[valid - 1];
    float row_gate[StateTile::ValueFragments][StateGateRows::Count];
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int half = 0; half < StateGateRows::Count; ++half) {
        int const row = StateTile::value_row(warp, r) + StateGateRows::row(lane, half);
        row_gate[r][half] = expf(last - sm.g[row]);
      }
    }
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = StateTile::value_row(warp, r) + rc.row;
        unsigned const at = V::offset(row, StateTile::column(warp) + rc.col);
        float const x = row < valid ? float(sm.u[at]) - value[r][s] : 0.0f;
        sm.u[at] = BF16(x);
        sm.scaled_v[at] = BF16(x * row_gate[r][StateGateRows::half(s)]);
      }
    }
    // VALUES_READY does not wait for the unrelated next W.
    __syncthreads();
    V::publish<StateTile::Threads>(sm.u, ws.vnew + tile_offset(group) + v0, Dim);
    float const decay = expf(last);
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) state[k][s] *= decay;
    }
    // UPDATE: exact admitted order; next W remains in flight.
    CUTE_UNROLL
    for (int r = 0; r < Chunk; r += 16) {
      uint32_t value_operand[4];
      V::load<true>(sm.scaled_v, r, StateTile::column(warp), value_operand);
      CUTE_UNROLL
      for (int k = 0; k < StateTile::KFragments; ++k) {
        uint32_t key[4];
        W::load<true>(sm.k, r, StateTile::k_row(warp, k), key);
        bf16_mma(state[k], key, value_operand);
      }
    }
    // No redundant end barrier: W_READY precedes every next K/U overwrite.
  }
  if (final) {
    // Last iteration did not issue NEXT_W; the W/final_h union is now dead.
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.final_h[(StateTile::k_row(warp, k) + rc.row) * ValueTile +
                    StateTile::column(warp) + rc.col] = state[k][s];
      }
    }
    __syncthreads();
    state_publish_fp32<Dim, ValueTile, StateTile::Threads>(
        sm.final_h, final + int64_t(bh) * Dim * Dim + v0, Dim);
  }
}
}  // namespace gdn_qsa::wy::state_pipeline

namespace gdn_qsa::wy {
int configure_state_pipeline() {
  return int(hggcFuncSetAttribute(state_pipeline::gdn_wy_state_pipeline,
      hggcFuncAttributeMaxDynamicSharedMemorySize, state_pipeline::Plan::SharedBytes));
}
int launch_state_pipeline(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream) {
  using namespace state_pipeline;
  unsigned const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));
  gdn_wy_state_pipeline<<<grid, Plan::Threads, Plan::SharedBytes, stream>>>(p, ws, final);
  return int(hggcGetLastError());
}
}  // namespace gdn_qsa::wy
