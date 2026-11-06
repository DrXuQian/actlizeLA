#include "common.cuh"
#include "gdn_qsa/ppu/wy_solve_static.cuh"

namespace actlize_la::ppu15 {
namespace {
struct SolveStorage {
  union {
    alignas(128) BF16 key[Chunk * Dim];
    alignas(128) float temporary[4][256];
    alignas(128) BF16 result[Chunk * Chunk];
  };
  alignas(128) float lower[Chunk * Chunk], inverse[Chunk * Chunk];
  float prefix[Chunk], beta[Chunk];
};
static_assert(sizeof(SolveStorage) == 49664);

__global__ void __launch_bounds__(64) prefix_kernel(Inputs p, Workspace ws) {
  __shared__ float prefix[Chunk];
  int const tid = int(threadIdx.x), lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks(), bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int64_t const at = (int64_t(b) * p.shape.sequence + first + min(tid, valid - 1)) * p.shape.value_heads + h;
  float g = tid < valid ? gate(p, at) : 0.0f;
  // Preserve the retained path's exact two 32-lane scans followed by a carry.
  CUTE_UNROLL
  for (int step = 1; step < 32; step *= 2) {
    float const peer = __shfl_up_sync(0xffffffffu, g, step);
    if (lane >= step) g += peer;
  }
  prefix[tid] = g;
  __syncthreads();
  if (tid >= 32) g += prefix[31];
  ws.gates[p.shape.group(b, h, ct) * Chunk + tid] = g;
}

__global__ void __launch_bounds__(128) solve_kernel(Inputs p, Workspace ws) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<SolveStorage*>(storage);
  int const tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks(), bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int64_t const group = p.shape.group(b, h, ct);
  using Key = Tile<Chunk, Dim>;
  Key::stage(sm.key, p.k + p.shape.input(b, first, p.shape.q_head(h), p.shape.q_heads),
             p.shape.q_heads * Dim, valid);
  if (tid < Chunk) {
    sm.prefix[tid] = ws.gates[group * Chunk + tid];
    sm.beta[tid] = tid < valid
        ? float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h]) : 0.0f;
  }
  commit_wait();
  for (int tile = warp; tile < 16; tile += 4) {
    int const br = tile / 4, bc = tile % 4;
    if (bc > br) continue;
    float acc[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t a[4], breg[4];
      Key::load<true>(sm.key, br * 16, k, a);
      Key::load<false>(sm.key, bc * 16, k, breg);
      bf16_mma(acc, a, breg);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const r = br * 16 + rc.row, c = bc * 16 + rc.col;
      sm.lower[r * Chunk + c] = r > c ? acc[s] * sm.beta[r] * expf(sm.prefix[r] - sm.prefix[c]) : 0.0f;
    }
  }
  __syncthreads();  // all key readers retire before union reuse
  float column[16];
  int const diagonal = warp * 16 * (Chunk + 1);
  gdn_qsa::wy::solve_static::diagonal(sm.lower + diagonal, sm.inverse + diagonal, lane, column);
  __syncthreads();
  #pragma unroll 1
  for (int gap = 1; gap < 4; ++gap) {
    int const br = warp + gap, bc = warp;
    if (br < 4) {
      float acc[8] = {};
      #pragma unroll 1
      for (int k = bc; k < br; ++k)
        tf32_product(acc, sm.lower + br * 16 * Chunk + k * 16, Chunk,
                     sm.inverse + k * 16 * Chunk + bc * 16, Chunk);
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.temporary[warp][rc.row * 16 + rc.col] = acc[s];
      }
      __syncwarp();
      float merged[8] = {};
      tf32_product(merged, sm.inverse + br * 16 * Chunk + br * 16, Chunk, sm.temporary[warp], 16);
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.inverse[(br * 16 + rc.row) * Chunk + bc * 16 + rc.col] = -merged[s];
      }
    }
    __syncthreads();
  }
  for (int i = tid; i < Chunk * Chunk; i += 128)
    sm.result[i] = BF16(i / Chunk >= i % Chunk ? sm.inverse[i] : 0.0f);
  __syncthreads();
  CUTE_UNROLL
  for (int it = 0; it < Chunk * Chunk / (128 * 8); ++it) {
    int const i = (tid + it * 128) * 8;
    *reinterpret_cast<uint4*>(ws.inverse + inverse_base(group) + i) =
        *reinterpret_cast<uint4 const*>(sm.result + i);
  }
}
}  // namespace

int configure_prepare() {
  return int(hggcFuncSetAttribute(solve_kernel, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(SolveStorage)));
}
int prepare(Inputs p, Workspace ws, hggcStream_t stream) {
  prefix_kernel<<<unsigned(p.shape.groups()), 64, 0, stream>>>(p, ws);
  auto rc = hggcGetLastError();
  if (rc) return int(rc);
  solve_kernel<<<unsigned(p.shape.groups()), 128, sizeof(SolveStorage), stream>>>(p, ws);
  return int(hggcGetLastError());
}
}  // namespace actlize_la::ppu15
