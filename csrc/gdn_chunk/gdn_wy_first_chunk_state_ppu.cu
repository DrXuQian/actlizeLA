// Absent-initial first-chunk specialization. All defined workspace writes remain.
// The source gate permits only zero initialization and guarded first-chunk KH.
// Prefix/solve and all nonzero-history recurrence terms remain unchanged.
#include "gdn_wy_common.cuh"
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_gate_coefficients.cuh"
#include "gdn_qsa/ppu/wy_first_chunk.hpp"

namespace gdn_qsa::wy::first_chunk {
using namespace residual_warps8_hvlayout;
using residual_warps8_hvlayout::StateTile;

__global__ void __launch_bounds__(Plan::Threads)
gdn_wy_residual_first_chunk_state(Inputs p, Workspace ws, float* final) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& shared = *reinterpret_cast<gate_cache::Storage*>(storage);
  auto& sm = shared.matrices;
  auto& coefficients = shared.coefficients;
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
  unsigned const h_store_base = Snapshot::producer_base(warp, lane);
  unsigned const b_store_base = BIntermediate::producer_base(warp, lane);
  int const slice = int(blockIdx.x) % (Dim / ValueTile);
  int const bh = int(blockIdx.x) / (Dim / ValueTile);
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const v0 = slice * ValueTile, qh = p.shape.q_head(h);
  float state[StateTile::KFragments][8] = {};
  #pragma unroll 1
  for (int ct = 0; ct < p.shape.chunks(); ++ct) {
    int64_t const group = p.shape.group(b, h, ct);
    int const first = ct * Chunk, valid = Chunk;
    Key::stage(sm.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
               p.shape.q_heads * Dim, valid);
    Inverse::stage(sm.inverse, ws.w + Plan::inverse_base(group), Chunk, Chunk);
    Value::stage(sm.value, p.v + p.shape.input(b, first, h, p.shape.value_heads) + v0,
                 p.shape.value_heads * Dim, valid);
    if (tid < Chunk) {
      // No cross-thread read before INPUTS_READY: both inputs come directly
      // from the already completed prefix kernel, including initialized tails.
      coefficients.publish(tid, ws.gates[group * Chunk + tid],
                           ws.gates[group * Chunk + valid - 1]);
      sm.beta[tid] = float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h]);
    }
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.snapshot[Snapshot::producer_offset(h_store_base, k, s)] = BF16(state[k][s]);
      }
    }
    commit_wait();  // INPUTS_READY: AIU completion + snapshot publication.
    Snapshot::publish<Plan::Threads>(sm.snapshot, ws.snapshots + Snapshot::workspace_offset(group, 0, v0), Dim);
    float kh[StateTile::ValueFragments][8] = {};
    if (has_history(ct)) {
      CUTE_UNROLL
      for (int k = 0; k < Dim; k += 16) {
        uint32_t hs[4];
        Snapshot::load(sm.snapshot, k, StateTile::column(warp), hs);
        CUTE_UNROLL
        for (int r = 0; r < StateTile::ValueFragments; ++r) {
          uint32_t key[4];
          Key::load(sm.k, StateTile::value_row(warp, r), k, key);
          bf16_mma(kh[r], key, hs);
        }
      }
    }
    float row_decay[StateTile::ValueFragments][StateGateRows::Count];
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      float factor[StateGateRows::Count], beta[StateGateRows::Count];
      CUTE_UNROLL
      for (int half = 0; half < StateGateRows::Count; ++half) {
        int const row = StateTile::value_row(warp, r) + StateGateRows::row(lane, half);
        factor[half] = coefficients.prefix[row];
        beta[half] = sm.beta[row];
        row_decay[r][half] = coefficients.relative[row];
      }
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = StateTile::value_row(warp, r) + rc.row;
        unsigned const at = Value::offset(row, StateTile::column(warp) + rc.col);
        int const half = StateGateRows::half(s);
        // Deliberate new rounding boundary. No exp(-prefix) / growing gate
        // factor: nonpositive log gates cannot overflow on strong decay.
        float const difference = float(sm.value[at]) - factor[half] * kh[r][s];
        sm.residual[BIntermediate::producer_offset(b_store_base, r, s)] = BF16(beta[half] * difference);
      }
    }
    __syncthreads();  // RESIDUAL_READY: P@R ready; all original input-V readers retired.
    float value[StateTile::ValueFragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Chunk; k += 16) {
      uint32_t residual[4];
      BIntermediate::load(sm.residual, k, StateTile::column(warp), residual);
      CUTE_UNROLL
      for (int r = 0; r < StateTile::ValueFragments; ++r) {
        uint32_t inverse[4];
        Inverse::load(sm.inverse, StateTile::value_row(warp, r), k, inverse);
        bf16_mma(value[r], inverse, residual);
      }
    }
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = StateTile::value_row(warp, r) + rc.row;
        float const x = value[r][s];
        sm.value[PublishedValue::producer_offset(b_store_base, r, s)] = BF16(x);
        sm.scaled[BIntermediate::producer_offset(b_store_base, r, s)] = BF16(x * row_decay[r][StateGateRows::half(s)]);
      }
    }
    __syncthreads();  // VALUES_READY: BF16 output and scaledV have different rounding.
    PublishedValue::publish<Plan::Threads>(sm.value, ws.vnew + PublishedValue::workspace_offset(group, 0, v0));
    float const decay = coefficients.prefix[valid - 1];
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) state[k][s] *= decay;
    }
    CUTE_UNROLL
    for (int r = 0; r < Chunk; r += 16) {
      uint32_t scaled[4];
      BIntermediate::load(sm.scaled, r, StateTile::column(warp), scaled);
      CUTE_UNROLL
      for (int k = 0; k < StateTile::KFragments; ++k) {
        uint32_t key[4];
        Key::load<true>(sm.k, r, StateTile::k_row(warp, k), key);
        bf16_mma(state[k], key, scaled);
      }
    }
    __syncthreads();  // RETIRE: all old readers before next AIU/shared overwrite.
  }
  if (final) {
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
    state_publish_fp32<Dim, ValueTile, Plan::Threads>(
        sm.final_h, final + int64_t(bh) * Dim * Dim + v0, Dim);
  }
}
// Keep the host composition separate from this device body.
int configure_state() {
  return int(hggcFuncSetAttribute(gdn_wy_residual_first_chunk_state,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(gate_cache::Storage)));
}
int launch_state(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream) {
  unsigned const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));
  gdn_wy_residual_first_chunk_state<<<grid, Plan::Threads, sizeof(gate_cache::Storage), stream>>>(p, ws, final);
  return int(hggcGetLastError());
}
}  // namespace gdn_qsa::wy::first_chunk
