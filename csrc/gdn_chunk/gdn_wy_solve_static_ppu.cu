// Isolated static diagonal indexing; the source gate binds every other
// solve statement to gdn_wy_split_solve. Precision and layouts are unchanged.
#include "gdn_wy_common.cuh"
#include "gdn_qsa/ppu/wy_aiu.cuh"
#include "gdn_qsa/ppu/wy_split_prepare.cuh"
#include "gdn_qsa/ppu/wy_solve_static.cuh"

namespace gdn_qsa::wy::solve_static {
using Key = aiu::Tile<Chunk, Dim>;
using split_prepare::Plan;
struct SolveStorage {
  union {
    alignas(128) BF16 key[Chunk * Dim];
    alignas(128) float temp[4][16 * 16];
    alignas(128) BF16 result[Chunk * Chunk];
  };
  alignas(128) float lower[Chunk * Chunk], inverse[Chunk * Chunk];
  float prefix[Chunk], beta[Chunk];
};
static_assert(sizeof(SolveStorage) == 49664);

__global__ void __launch_bounds__(Plan::SolveThreads)
gdn_wy_split_solve_static(Inputs p, Workspace ws) {
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
  unsigned const diagonal_base = warp * 16 * (Chunk + 1);
  solve_static::diagonal(sm.lower + diagonal_base, sm.inverse + diagonal_base, lane, column);
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


int configure() {
  return int(hggcFuncSetAttribute(gdn_wy_split_solve_static,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(SolveStorage)));
}
int launch_inverse(Inputs p, Workspace ws, gdn_arch::Stream stream) {
  int const rc = launch_split_prefix(p, ws, stream);
  if (rc) return rc;
  gdn_wy_split_solve_static<<<unsigned(p.shape.groups()), Plan::SolveThreads,
                             sizeof(SolveStorage), stream>>>(p, ws);
  return int(hggcGetLastError());
}
}  // namespace gdn_qsa::wy::solve_static

namespace gdn_qsa::wy::residual_warps8_hvlayout {
int configure_hvlayout_state();
int launch_hvlayout_state(Inputs, Workspace, float*, gdn_arch::Stream);
int configure_hvlayout_output();
int launch_hvlayout_output(Inputs, Workspace, BF16*, gdn_arch::Stream);
}

extern "C" int gdn_wy_forward_residual_solve_static(
    void const* q, void const* k, void const* v, void const* g, void const* beta, float const* initial,
    void* output, float* final, void* inverse, void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, gdn_arch::Stream stream) {
  using namespace gdn_qsa::wy;
  using namespace residual_warps8_hvlayout;
  using solve_static::Key;
  if (batch <= 0 || sequence <= 0 || q_heads <= 0 || value_heads <= 0 || value_heads % q_heads ||
      !Key::admitted_stride(int64_t(q_heads) * Dim) ||
      !Key::admitted_stride(int64_t(value_heads) * Dim) || inverse == snapshots)
    return int(hggcErrorInvalidValue);
  Inputs p{static_cast<BF16 const*>(q), static_cast<BF16 const*>(k), static_cast<BF16 const*>(v),
           static_cast<BF16 const*>(beta), g, initial, gate_fp32, {batch, sequence, q_heads, value_heads}};
  Workspace ws{static_cast<BF16*>(inverse), nullptr, static_cast<BF16*>(snapshots),
               static_cast<BF16*>(vnew), gates};
  int rc = solve_static::configure();
  if (rc) return rc;
  rc = configure_hvlayout_state();
  if (rc) return rc;
  rc = configure_hvlayout_output();
  if (rc) return rc;
  Workspace inverse_ws = ws;
  inverse_ws.snapshots = ws.w;  // separate inverse and live H lifetimes, as in HV
  rc = solve_static::launch_inverse(p, inverse_ws, stream);
  if (rc) return rc;
  rc = launch_hvlayout_state(p, ws, final, stream);
  if (rc) return rc;
  return launch_hvlayout_output(p, ws, static_cast<BF16*>(output), stream);
}
