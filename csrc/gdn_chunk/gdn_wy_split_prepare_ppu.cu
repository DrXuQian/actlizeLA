// Separate resource budgets for prefix, KKT+solve and WU. Forward only.
// This is NOT the older tiled-prepare experiment: state/output stay AIU13808,
// the solve keeps all TF32 residual terms, and WU has its own256-thread CTA.
#include "gdn_wy_common.cuh"
#include "gdn_qsa/ppu/wy_aiu.cuh"
#include "gdn_qsa/ppu/wy_split_prepare.cuh"

namespace gdn_qsa::wy::split_prepare {
using Key = aiu::Tile<Chunk, Dim>;
using Inverse = aiu::Tile<Chunk, Chunk>;

struct SolveStorage {
  union {
    alignas(128) BF16 key[Chunk * Dim];
    alignas(128) float temp[4][16 * 16];
    alignas(128) BF16 result[Chunk * Chunk];
  };
  alignas(128) float lower[Chunk * Chunk], inverse[Chunk * Chunk];
  float prefix[Chunk], beta[Chunk];
};
struct WUStorage {
  alignas(128) BF16 inverse[Chunk * Chunk];
  // K/V readers finish before these planes become the W/U publication tiles.
  alignas(128) BF16 key[Chunk * Dim], value[Chunk * Dim];
  float factor[Chunk], beta[Chunk];
};
static_assert(sizeof(SolveStorage) == 49664);
static_assert(sizeof(WUStorage) == 41472);

__global__ void __launch_bounds__(Plan::PrefixThreads)
gdn_wy_split_prefix(Inputs p, Workspace ws) {
  __shared__ float prefix[Chunk];
  unsigned const tid = unsigned(threadIdx.x), lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks();
  int const bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int64_t const group = p.shape.group(b, h, ct);
  int64_t const at = (int64_t(b) * p.shape.sequence + first + min(int(tid), valid - 1))
                    * p.shape.value_heads + h;
  float g = tid < unsigned(valid) ? gate(p, at) : 0.0f;
  // Exactly the original two independent32-lane scans, followed by one carry.
  CUTE_UNROLL
  for (int offset = 1; offset < 32; offset *= 2) {
    float const other = __shfl_up_sync(0xffffffffu, g, offset);
    g = prefix_step(g, other, lane, unsigned(offset));
  }
  prefix[tid] = g;
  __syncthreads();
  g = prefix_carry(g, prefix[31], tid);
  ws.gates[group * Chunk + tid] = g;
}

__global__ void __launch_bounds__(Plan::SolveThreads)
gdn_wy_split_solve(Inputs p, Workspace ws) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<SolveStorage*>(storage);
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks();
  int const bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int64_t const group = p.shape.group(b, h, ct);
  Key::stage(sm.key, p.k + p.shape.input(b, first, p.shape.q_head(h), p.shape.q_heads),
             p.shape.q_heads * Dim, valid);
  if (tid < Chunk) {
    sm.prefix[tid] = ws.gates[group * Chunk + tid];
    int64_t const at = (int64_t(b) * p.shape.sequence + first + min(int(tid), valid - 1))
                      * p.shape.value_heads + h;
    sm.beta[tid] = tid < unsigned(valid) ? float(p.beta[at]) : 0.0f;
  }
  commit_wait();
  // Same ten products, K-atom order, row weighting and FP32 diagonal solve
  // as prepare_inverse. No BF16 lower matrix / no TF32 residual removal.
  for (int tile = int(warp); tile < 16; tile += 4) {
    int const br = tile / 4, bc = tile % 4;
    if (bc > br) continue;
    float acc[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t a[4], bt[4];
      Key::load(sm.key, br * 16, k, a);
      Key::load(sm.key, bc * 16, k, bt);
      bf16_mma(acc, a, bt);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const r = br * 16 + rc.row, c = bc * 16 + rc.col;
      sm.lower[r * Chunk + c] = r > c
          ? acc[s] * sm.beta[r] * expf(sm.prefix[r] - sm.prefix[c]) : 0.0f;
    }
  }
  __syncthreads();  // all key readers retire before union scratch reuse
  float column[16];
  CUTE_UNROLL
  for (int r = 0; r < 16; ++r) {
    float x = r == int(lane % 16) ? 1.0f : 0.0f;
    CUTE_UNROLL
    for (int k = 0; k < r; ++k)
      x -= sm.lower[(warp * 16 + r) * Chunk + warp * 16 + k] * column[k];
    column[r] = x;
    if (lane < 16) sm.inverse[(warp * 16 + r) * Chunk + warp * 16 + lane] = x;
  }
  __syncthreads();
  #pragma unroll 1
  for (int gap = 1; gap < 4; ++gap) {
    int const br = int(warp) + gap, bc = int(warp);
    if (br < 4) {
      float acc[8] = {};
      #pragma unroll 1
      for (int k = bc; k < br; ++k)
        tf32_product(acc, sm.lower + br * 16 * Chunk + k * 16, Chunk,
                     sm.inverse + k * 16 * Chunk + bc * 16, Chunk);
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.temp[warp][rc.row * 16 + rc.col] = acc[s];
      }
      __syncwarp();
      float merged[8] = {};
      tf32_product(merged, sm.inverse + br * 16 * Chunk + br * 16, Chunk, sm.temp[warp], 16);
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.inverse[(br * 16 + rc.row) * Chunk + bc * 16 + rc.col] = -merged[s];
      }
    }
    __syncthreads();
  }
  // Temp readers retired. Materialize every entry, including upper-triangle
  // zeros; no memset/fill kernel and no read of uninitialized upper inverse.
  for (unsigned i = tid; i < Plan::InverseElements; i += Plan::SolveThreads)
    sm.result[i] = BF16(i / Chunk >= i % Chunk ? sm.inverse[i] : 0.0f);
  __syncthreads();
  CUTE_UNROLL
  for (unsigned it = 0; it < Plan::InverseIterations; ++it) {
    unsigned const i = (tid + it * Plan::SolveThreads) * Plan::VectorElements;
    uint4 const packed = *reinterpret_cast<uint4 const*>(sm.result + i);
    cutlass::arch::global_store<uint4, 16>(packed,
        ws.snapshots + Plan::inverse_base(group) + i, true);
  }
}

__global__ void __launch_bounds__(Plan::WUThreads)
gdn_wy_split_wu(Inputs p, Workspace ws) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<WUStorage*>(storage);
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks();
  int const bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int64_t const group = p.shape.group(b, h, ct);
  Inverse::stage(sm.inverse, ws.snapshots + Plan::inverse_base(group), Chunk, Chunk);
  Key::stage(sm.key, p.k + p.shape.input(b, first, p.shape.q_head(h), p.shape.q_heads),
             p.shape.q_heads * Dim, valid);
  Key::stage(sm.value, p.v + p.shape.input(b, first, h, p.shape.value_heads),
             p.shape.value_heads * Dim, valid);
  if (tid < Chunk) {
    sm.factor[tid] = expf(ws.gates[group * Chunk + tid]);
    int64_t const at = (int64_t(b) * p.shape.sequence + first + min(int(tid), valid - 1))
                      * p.shape.value_heads + h;
    sm.beta[tid] = tid < unsigned(valid) ? float(p.beta[at]) : 0.0f;
  }
  commit_wait();
  CUTE_UNROLL
  for (unsigned it = 0; it < Plan::ValueIterations; ++it) {
    unsigned const i = Plan::value_vector(tid, it), row = i / Dim, col = i % Dim;
    unsigned const at = Key::offset(row, col);
    // One contiguous physical vector. Preserve beta THEN exp, BF16 boundary.
    CUTE_UNROLL
    for (unsigned j = 0; j < Plan::VectorElements; ++j) {
      sm.key[at + j] = condition_key(sm.key[at + j], sm.beta[row], sm.factor[row]);
      sm.value[at + j] = condition_value(sm.value[at + j], sm.beta[row]);
    }
  }
  __syncthreads();
  float w[Plan::WUFragments][8] = {}, u[Plan::WUFragments][8] = {};
  CUTE_UNROLL
  for (int k = 0; k < Chunk; k += 16) {
    CUTE_UNROLL
    for (unsigned r = 0; r < 2; ++r) {
      uint32_t a[4];
      Inverse::load(sm.inverse, Plan::wu_row(warp, r * 2), k, a);
      CUTE_UNROLL
      for (unsigned c = 0; c < 2; ++c) {
        unsigned const f = r * 2 + c;
        uint32_t bk[4], bv[4];
        Key::load<true>(sm.key, k, Plan::wu_col(warp, f), bk);
        Key::load<true>(sm.value, k, Plan::wu_col(warp, f), bv);
        bf16_mma(w[f], a, bk); bf16_mma(u[f], a, bv);
      }
    }
  }
  __syncthreads();  // all key/value readers retire before publication exchange
  CUTE_UNROLL
  for (unsigned f = 0; f < Plan::WUFragments; ++f) {
    CUTE_UNROLL
    for (unsigned s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      unsigned const at = Key::offset(Plan::wu_row(warp, f) + rc.row, Plan::wu_col(warp, f) + rc.col);
      sm.key[at] = BF16(w[f][s]); sm.value[at] = BF16(u[f][s]);
    }
  }
  __syncthreads();
  Key::publish<Plan::WUThreads>(sm.key, ws.w + tile_offset(group), Dim);
  Key::publish<Plan::WUThreads>(sm.value, ws.u + tile_offset(group), Dim);
}

}  // namespace gdn_qsa::wy::split_prepare

namespace gdn_qsa::wy {
// Host-only reuse for isolated solve experiments; existing launchers and
// all three admitted device bodies remain unchanged.
int launch_split_prefix(Inputs p, Workspace ws, gdn_arch::Stream stream) {
  using namespace split_prepare;
  gdn_wy_split_prefix<<<unsigned(p.shape.groups()), Plan::PrefixThreads, 0, stream>>>(p, ws);
  return int(hggcGetLastError());
}
int configure_split_prepare() {
  using namespace split_prepare;
  int rc = int(hggcFuncSetAttribute(gdn_wy_split_solve,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(SolveStorage)));
  if (rc) return rc;
  return int(hggcFuncSetAttribute(gdn_wy_split_wu,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(WUStorage)));
}
int launch_split_inverse(Inputs p, Workspace ws, gdn_arch::Stream stream) {
  using namespace split_prepare;
  unsigned const grid = unsigned(p.shape.groups());
  gdn_wy_split_prefix<<<grid, Plan::PrefixThreads, 0, stream>>>(p, ws);
  int rc = int(hggcGetLastError());
  if (rc) return rc;
  gdn_wy_split_solve<<<grid, Plan::SolveThreads, sizeof(SolveStorage), stream>>>(p, ws);
  return int(hggcGetLastError());
}
int launch_split_prepare(Inputs p, Workspace ws, gdn_arch::Stream stream) {
  using namespace split_prepare;
  unsigned const grid = unsigned(p.shape.groups());
  int const rc = launch_split_inverse(p, ws, stream);
  if (rc) return rc;
  gdn_wy_split_wu<<<grid, Plan::WUThreads, sizeof(WUStorage), stream>>>(p, ws);
  return int(hggcGetLastError());
}
}  // namespace gdn_qsa::wy
