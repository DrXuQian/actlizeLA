// Delivery-only arm of the admitted residual algorithm. Directly place the
// two non-published intermediates in B orientation; no additional data move.
#include "gdn_wy_common.cuh"
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_residual_blayout.cuh"

namespace gdn_qsa::wy::residual_blayout {

__global__ void __launch_bounds__(Plan::Threads)
gdn_wy_residual_blayout_state(Inputs p, Workspace ws, float* final) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<Storage*>(storage);
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
  unsigned const b_store_base = BIntermediate::producer_base(warp,lane);
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
  #pragma unroll 1
  for (int ct = 0; ct < p.shape.chunks(); ++ct) {
    int64_t const group = p.shape.group(b, h, ct);
    int const first = ct * Chunk, valid = Plan::valid(ct, p.shape.sequence);
    Key::stage(sm.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
               p.shape.q_heads * Dim, valid);
    Inverse::stage(sm.inverse, ws.w + Plan::inverse_base(group), Chunk, Chunk);
    Value::stage(sm.value, p.v + p.shape.input(b, first, h, p.shape.value_heads) + v0,
                 p.shape.value_heads * Dim, valid);
    if (tid < Chunk) {
      sm.gates[tid] = ws.gates[group * Chunk + tid];
      sm.beta[tid] = tid < unsigned(valid)
          ? float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h]) : 0.0f;
    }
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.snapshot[Snapshot::offset(StateTile::k_row(warp, k) + rc.row,
                                      StateTile::column(warp) + rc.col)] = BF16(state[k][s]);
      }
    }
    commit_wait();  // INPUTS_READY: AIU completion + snapshot publication.
    Snapshot::publish<Plan::Threads>(sm.snapshot, ws.snapshots + state_offset(group) + v0, Dim);
    float kh[StateTile::ValueFragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t hs[4];
      Snapshot::load<true>(sm.snapshot, k, StateTile::column(warp), hs);
      CUTE_UNROLL
      for (int r = 0; r < StateTile::ValueFragments; ++r) {
        uint32_t key[4];
        Key::load(sm.k, StateTile::value_row(warp, r), k, key);
        bf16_mma(kh[r], key, hs);
      }
    }
    float const last = sm.gates[valid - 1];
    float row_decay[StateTile::ValueFragments][StateGateRows::Count];
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      float factor[StateGateRows::Count], beta[StateGateRows::Count];
      CUTE_UNROLL
      for (int half = 0; half < StateGateRows::Count; ++half) {
        int const row = StateTile::value_row(warp, r) + StateGateRows::row(lane, half);
        factor[half] = expf(sm.gates[row]);
        beta[half] = sm.beta[row];
        row_decay[r][half] = expf(last - sm.gates[row]);
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
        sm.residual[BIntermediate::producer_offset(b_store_base,r,s)] = BF16(row < valid ? beta[half] * difference : 0.0f);
      }
    }
    __syncthreads();  // RESIDUAL_READY: every P@R reader sees all K rows.
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
        unsigned const at = Value::offset(row, StateTile::column(warp) + rc.col);
        float const x = row < valid ? value[r][s] : 0.0f;
        sm.value[at] = BF16(x);
        sm.scaled[BIntermediate::producer_offset(b_store_base,r,s)] = BF16(x * row_decay[r][StateGateRows::half(s)]);
      }
    }
    __syncthreads();  // VALUES_READY: BF16 output and scaledV have different rounding.
    Value::publish<Plan::Threads>(sm.value, ws.vnew + tile_offset(group) + v0, Dim);
    float const decay = expf(last);
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
}  // namespace gdn_qsa::wy::residual_blayout

extern "C" int gdn_wy_forward_residual_blayout(
    void const* q, void const* k, void const* v, void const* g, void const* beta, float const* initial,
    void* output, float* final, void* inverse, void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, gdn_arch::Stream stream) {
  using namespace gdn_qsa::wy;
  using namespace residual_blayout;
  if (batch <= 0 || sequence <= 0 || q_heads <= 0 || value_heads <= 0 || value_heads % q_heads ||
      !Key::admitted_stride(int64_t(q_heads) * Dim) ||
      !Key::admitted_stride(int64_t(value_heads) * Dim) || inverse == snapshots)
    return int(hggcErrorInvalidValue);
  Inputs p{static_cast<BF16 const*>(q), static_cast<BF16 const*>(k), static_cast<BF16 const*>(v),
           static_cast<BF16 const*>(beta), g, initial, gate_fp32, {batch, sequence, q_heads, value_heads}};
  Workspace ws{static_cast<BF16*>(inverse), nullptr, static_cast<BF16*>(snapshots),
               static_cast<BF16*>(vnew), gates};
  int rc = configure_split_prepare();
  if (rc) return rc;
  rc = int(hggcFuncSetAttribute(gdn_wy_residual_blayout_state,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage)));
  if (rc) return rc;
  rc = configure_aiu_output();
  if (rc) return rc;
  Workspace inverse_ws = ws;
  inverse_ws.snapshots = ws.w;  // old solve writes its padded pitch, NOT the live H workspace.
  rc = launch_split_inverse(p, inverse_ws, stream);
  if (rc) return rc;
  unsigned const grid = unsigned(int64_t(batch) * value_heads * (Dim / ValueTile));
  gdn_wy_residual_blayout_state<<<grid, Plan::Threads, sizeof(Storage), stream>>>(p, ws, final);
  rc = int(hggcGetLastError());
  if (rc) return rc;
  return launch_aiu_output(p, ws, static_cast<BF16*>(output), stream);
}
