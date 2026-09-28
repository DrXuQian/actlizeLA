// Paired H shared/snapshot/output layout; old control TU remains unchanged.
// Paired private Vnew publication and output reader on the measured H-layout.
// Gated inverse, residual math/rounding and public output/state are unchanged.
#include "gdn_wy_common.cuh"
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_residual_warps8_hvlayout.cuh"

namespace gdn_qsa::wy::residual_warps8_hvlayout {

__global__ void __launch_bounds__(Plan::Threads)
gdn_wy_residual_warps8_hvlayout_state(Inputs p, Workspace ws, float* final) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<Storage*>(storage);
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
  unsigned const h_store_base = Snapshot::producer_base(warp, lane);
  unsigned const b_store_base = BIntermediate::producer_base(warp, lane);
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
        sm.snapshot[Snapshot::producer_offset(h_store_base, k, s)] = BF16(state[k][s]);
      }
    }
    commit_wait();  // INPUTS_READY: AIU completion + snapshot publication.
    Snapshot::publish<Plan::Threads>(sm.snapshot, ws.snapshots + Snapshot::workspace_offset(group, 0, v0), Dim);
    float kh[StateTile::ValueFragments][8] = {};
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
        sm.residual[BIntermediate::producer_offset(b_store_base, r, s)] = BF16(row < valid ? beta[half] * difference : 0.0f);
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
        float const x = row < valid ? value[r][s] : 0.0f;
        sm.value[PublishedValue::producer_offset(b_store_base, r, s)] = BF16(x);
        sm.scaled[BIntermediate::producer_offset(b_store_base, r, s)] = BF16(x * row_decay[r][StateGateRows::half(s)]);
      }
    }
    __syncthreads();  // VALUES_READY: BF16 output and scaledV have different rounding.
    PublishedValue::publish<Plan::Threads>(sm.value, ws.vnew + PublishedValue::workspace_offset(group, 0, v0));
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
__global__ void __launch_bounds__(OutputTile::Threads)
gdn_wy_residual_warps8_hvlayout_output(Inputs p, Workspace ws, BF16* output) {
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
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
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
          row >= col && row < valid ? attention[c][s] * expf(sm.g[row] - sm.g[col]) : 0.0f);
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

// Host-only entrypoints let a solve experiment reuse these exact device
// symbols, rather than copying or recompiling the state/output math.
int configure_hvlayout_state() {
  return int(hggcFuncSetAttribute(gdn_wy_residual_warps8_hvlayout_state,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage)));
}
int launch_hvlayout_state(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream) {
  unsigned const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));
  gdn_wy_residual_warps8_hvlayout_state<<<grid, Plan::Threads, sizeof(Storage), stream>>>(p, ws, final);
  return int(hggcGetLastError());
}
int configure_hvlayout_output() {
  return int(hggcFuncSetAttribute(gdn_wy_residual_warps8_hvlayout_output,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledOutputStorage)));
}
int launch_hvlayout_output(Inputs p, Workspace ws, BF16* output, gdn_arch::Stream stream) {
  gdn_wy_residual_warps8_hvlayout_output<<<unsigned(p.shape.groups()), OutputTile::Threads,
                                       sizeof(TiledOutputStorage), stream>>>(p, ws, output);
  return int(hggcGetLastError());
}

}  // namespace gdn_qsa::wy::residual_warps8_hvlayout

extern "C" int gdn_wy_forward_residual_warps8_hvlayout(
    void const* q, void const* k, void const* v, void const* g, void const* beta, float const* initial,
    void* output, float* final, void* inverse, void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, gdn_arch::Stream stream) {
  using namespace gdn_qsa::wy;
  using namespace residual_warps8_hvlayout;
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
  rc = int(hggcFuncSetAttribute(gdn_wy_residual_warps8_hvlayout_state,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage)));
  if (rc) return rc;
  rc = configure_hvlayout_output();
  if (rc) return rc;
  Workspace inverse_ws = ws;
  inverse_ws.snapshots = ws.w;  // old solve writes its padded pitch, NOT the live H workspace.
  rc = launch_split_inverse(p, inverse_ws, stream);
  if (rc) return rc;
  unsigned const grid = unsigned(int64_t(batch) * value_heads * (Dim / ValueTile));
  gdn_wy_residual_warps8_hvlayout_state<<<grid, Plan::Threads, sizeof(Storage), stream>>>(p, ws, final);
  rc = int(hggcGetLastError());
  if (rc) return rc;
  return launch_hvlayout_output(p, ws, static_cast<BF16*>(output), stream);
}
