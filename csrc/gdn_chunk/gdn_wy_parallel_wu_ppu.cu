// KDA counterfactual: chunk-parallel W/U, two products in the recurrent chain.
// NOT a delivery-only change: W/U introduce their own BF16 rounding boundaries.
// Built only with ALA_PPU10_PARALLEL_WU_EXPERIMENT; no default route is replaced.
#include "gdn_wy_common.cuh"
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_gate_coefficients.cuh"
#include "gdn_qsa/ppu/wy_parallel_wu.hpp"
#include "gdn_qsa/ppu/wy_split_prepare.cuh"
#include <climits>

namespace gdn_qsa::wy::solve_static {
int configure();
int launch_inverse_default(Inputs, Workspace, gdn_arch::Stream);
}
namespace gdn_qsa::wy::residual_warps8_hvlayout {
int configure_hvlayout_output();
int launch_hvlayout_output(Inputs, Workspace, BF16*, gdn_arch::Stream);
}

namespace gdn_qsa::wy::parallel_wu {
using namespace residual_warps8_hvlayout;
using residual_warps8_hvlayout::StateTile;
using WUPlan = split_prepare::Plan;
static_assert(WorkspacePlan::GroupElements == Dim * Dim && Chunk == 64);

struct WUStorage {
  alignas(128) BF16 inverse[Chunk * Chunk];
  alignas(128) BF16 key[Chunk * Dim], value[Chunk * Dim];
  float factor[Chunk], beta[Chunk];
};
static_assert(sizeof(WUStorage) == 41472);

// Reuse the retained split-WU arithmetic, atom order and AIU/SWZL transport.
// Only its output addressing/lifetime differs: padded per-group W/U replaces P.
__global__ void __launch_bounds__(WUPlan::WUThreads)
gdn_wy_parallel_wu_materialize(Inputs p, Workspace ws) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<WUStorage*>(storage);
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks();
  int const bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const first = ct * Chunk;
  int64_t const group = p.shape.group(b, h, ct);
  Inverse::stage(sm.inverse, ws.w + WorkspacePlan::w(group), Chunk, Chunk);
  Key::stage(sm.key, p.k + p.shape.input(b, first, p.shape.q_head(h), p.shape.q_heads),
             p.shape.q_heads * Dim, Chunk);
  Key::stage(sm.value, p.v + p.shape.input(b, first, h, p.shape.value_heads),
             p.shape.value_heads * Dim, Chunk);
  if (tid < Chunk) {
    sm.factor[tid] = expf(ws.gates[group * Chunk + tid]);
    sm.beta[tid] = float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h]);
  }
  commit_wait();  // P is fully resident before any global W/U overwrite.
  CUTE_UNROLL
  for (unsigned it = 0; it < WUPlan::ValueIterations; ++it) {
    unsigned const i = WUPlan::value_vector(tid, it), row = i / Dim, col = i % Dim;
    unsigned const at = Key::offset(row, col);
    CUTE_UNROLL
    for (unsigned j = 0; j < WUPlan::VectorElements; ++j) {
      sm.key[at + j] = split_prepare::condition_key(sm.key[at + j], sm.beta[row], sm.factor[row]);
      sm.value[at + j] = split_prepare::condition_value(sm.value[at + j], sm.beta[row]);
    }
  }
  __syncthreads();
  float w[WUPlan::WUFragments][8] = {}, u[WUPlan::WUFragments][8] = {};
  CUTE_UNROLL
  for (int k = 0; k < Chunk; k += 16) {
    CUTE_UNROLL
    for (unsigned r = 0; r < 2; ++r) {
      uint32_t a[4];
      Inverse::load(sm.inverse, WUPlan::wu_row(warp, r * 2), k, a);
      CUTE_UNROLL
      for (unsigned c = 0; c < 2; ++c) {
        unsigned const f = r * 2 + c;
        uint32_t bk[4], bv[4];
        Key::load<true>(sm.key, k, WUPlan::wu_col(warp, f), bk);
        Key::load<true>(sm.value, k, WUPlan::wu_col(warp, f), bv);
        bf16_mma(w[f], a, bk);
        bf16_mma(u[f], a, bv);
      }
    }
  }
  __syncthreads();  // Retire key/value readers before publication exchange.
  CUTE_UNROLL
  for (unsigned f = 0; f < WUPlan::WUFragments; ++f) {
    CUTE_UNROLL
    for (unsigned s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      unsigned const at = Key::offset(WUPlan::wu_row(warp, f) + rc.row, WUPlan::wu_col(warp, f) + rc.col);
      sm.key[at] = BF16(w[f][s]);
      sm.value[at] = BF16(u[f][s]);
    }
  }
  __syncthreads();
  Key::publish<WUPlan::WUThreads>(sm.key, ws.w + WorkspacePlan::w(group), Dim);
  Key::publish<WUPlan::WUThreads>(sm.value, ws.w + WorkspacePlan::u(group), Dim);
}

struct StateStorage {
  union {
    alignas(128) BF16 w[Chunk * Dim];
    alignas(128) float final_h[Dim * ValueTile];
  };
  alignas(128) BF16 key[Chunk * Dim];
  alignas(128) BF16 snapshot[Dim * ValueTile];
  alignas(128) BF16 value[Chunk * ValueTile], scaled[Chunk * ValueTile];
  gate_cache::Coefficients coefficients;
};
static_assert(sizeof(StateStorage) == 49664);

__global__ void __launch_bounds__(Plan::Threads)
gdn_wy_parallel_wu_state(Inputs p, Workspace ws, float* final) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<StateStorage*>(storage);
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
    int const first = ct * Chunk;
    Key::stage(sm.key, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
               p.shape.q_heads * Dim, Chunk);
    Key::stage(sm.w, ws.w + WorkspacePlan::w(group), Dim, Chunk);
    Value::stage(sm.value, ws.w + WorkspacePlan::u(group) + v0, Dim, Chunk);
    if (tid < Chunk)
      sm.coefficients.publish(tid, ws.gates[group * Chunk + tid], ws.gates[group * Chunk + Chunk - 1]);
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s)
        sm.snapshot[Snapshot::producer_offset(h_store_base, k, s)] = BF16(state[k][s]);
    }
    commit_wait();
    Snapshot::publish<Plan::Threads>(sm.snapshot, ws.snapshots + Snapshot::workspace_offset(group, 0, v0), Dim);
    float projected[StateTile::ValueFragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t hs[4];
      Snapshot::load(sm.snapshot, k, StateTile::column(warp), hs);
      CUTE_UNROLL
      for (int r = 0; r < StateTile::ValueFragments; ++r) {
        uint32_t w[4];
        Key::load(sm.w, StateTile::value_row(warp, r), k, w);
        bf16_mma(projected[r], w, hs);
      }
    }
    float value[StateTile::ValueFragments][8];
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = StateTile::value_row(warp, r) + rc.row;
        value[r][s] = float(sm.value[Value::offset(row, StateTile::column(warp) + rc.col)]) - projected[r][s];
      }
    }
    __syncthreads();  // All U readers retire before changing its physical layout.
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = StateTile::value_row(warp, r) + rc.row;
        float const x = value[r][s];
        sm.value[PublishedValue::producer_offset(b_store_base, r, s)] = BF16(x);
        sm.scaled[BIntermediate::producer_offset(b_store_base, r, s)] = BF16(x * sm.coefficients.relative[row]);
      }
    }
    __syncthreads();
    PublishedValue::publish<Plan::Threads>(sm.value, ws.vnew + PublishedValue::workspace_offset(group, 0, v0));
    float const decay = sm.coefficients.prefix[Chunk - 1];
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
        Key::load<true>(sm.key, r, StateTile::k_row(warp, k), key);
        bf16_mma(state[k], key, scaled);
      }
    }
    __syncthreads();  // Retire every reader before next chunk's AIU overwrite.
  }
  if (final) {
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.final_h[(StateTile::k_row(warp, k) + rc.row) * ValueTile + StateTile::column(warp) + rc.col] = state[k][s];
      }
    }
    __syncthreads();
    state_publish_fp32<Dim, ValueTile, Plan::Threads>(sm.final_h, final + int64_t(bh) * Dim * Dim + v0, Dim);
  }
}
}  // namespace gdn_qsa::wy::parallel_wu

extern "C" int actlize_la_ppu10_parallel_wu_configure() {
  using namespace gdn_qsa::wy;
  using namespace parallel_wu;
  if (int rc = solve_static::configure()) return rc;
  if (int rc = residual_warps8_hvlayout::configure_hvlayout_output()) return rc;
  if (int rc = hggcFuncSetAttribute(gdn_wy_parallel_wu_materialize,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(WUStorage))) return rc;
  return int(hggcFuncSetAttribute(gdn_wy_parallel_wu_state,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(StateStorage)));
}

extern "C" int actlize_la_ppu10_parallel_wu_forward_prepared(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* output, float* final, void* inverse,
    void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, void* opaque_stream) {
  using namespace gdn_qsa::wy;
  using namespace parallel_wu;
  if (!q || !k || !v || !g || !beta || !output || !inverse || !snapshots || !vnew || !gates ||
      inverse == snapshots || inverse == vnew || snapshots == vnew ||
      batch <= 0 || sequence <= 0 || sequence % Chunk || q_heads <= 0 || value_heads <= 0 ||
      value_heads % q_heads || q_heads > INT_MAX / Dim || value_heads > INT_MAX / Dim)
    return int(hggcErrorInvalidValue);
  Shape const shape{batch, sequence, q_heads, value_heads};
  if (int64_t(batch) * value_heads > INT_MAX / 4 || int64_t(batch) * value_heads > INT_MAX / shape.chunks())
    return int(hggcErrorInvalidValue);
  Inputs p{static_cast<BF16 const*>(q), static_cast<BF16 const*>(k), static_cast<BF16 const*>(v),
           static_cast<BF16 const*>(beta), g, initial, gate_fp32, shape};
  Workspace ws{static_cast<BF16*>(inverse), nullptr, static_cast<BF16*>(snapshots), static_cast<BF16*>(vnew), gates};
  auto stream = static_cast<gdn_arch::Stream>(opaque_stream);
  Workspace inverse_ws = ws;
  inverse_ws.snapshots = ws.w;
  if (int rc = solve_static::launch_inverse_default(p, inverse_ws, stream)) return rc;
  gdn_wy_parallel_wu_materialize<<<unsigned(shape.groups()), WUPlan::WUThreads, sizeof(WUStorage), stream>>>(p, ws);
  if (int rc = hggcGetLastError()) return rc;
  gdn_wy_parallel_wu_state<<<unsigned(int64_t(batch) * value_heads * 4), Plan::Threads, sizeof(StateStorage), stream>>>(p, ws, final);
  if (int rc = hggcGetLastError()) return rc;
  return residual_warps8_hvlayout::launch_hvlayout_output(p, ws, static_cast<BF16*>(output), stream);
}
