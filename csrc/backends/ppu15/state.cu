#include "common.cuh"

namespace actlize_la::ppu15 {
namespace {
using Key = Tile<Chunk, Dim>;
using Inverse = Tile<Chunk, Chunk>;
using Value = Tile<Chunk, ValueTile>;
using Snapshot = Tile<ValueTile, Dim>;  // physical [V,K], shared AND global
using BIntermediate = Tile<ValueTile, Chunk>;  // physical [V,T]
struct StateStorage {
  union {
    alignas(128) BF16 key[Chunk * Dim];
    alignas(128) float final_h[Dim * ValueTile];
  };
  alignas(128) BF16 inverse[Chunk * Chunk], snapshot[Dim * ValueTile];
  alignas(128) BF16 value[Chunk * ValueTile], residual[Chunk * ValueTile], scaled[Chunk * ValueTile];
  float beta[Chunk], prefix_exp[Chunk], relative_exp[Chunk];
};
static_assert(sizeof(StateStorage) == 45824);

// Full-C64 only removes predicates, not arithmetic. Tail T uses the same
// coefficient reuse and native operand layouts, with explicit zero padding.
template <bool FullChunks>
__global__ void __launch_bounds__(StateTile::Threads) state_kernel(Inputs p, Workspace ws, float* final) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<StateStorage*>(storage);
  int const tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const slice = int(blockIdx.x) % (Dim / ValueTile), bh = int(blockIdx.x) / (Dim / ValueTile);
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const v0 = slice * ValueTile, qh = p.shape.q_head(h), col = StateTile::col(warp);
  float hreg[StateTile::KFragments][8];
  CUTE_UNROLL
  for (int f = 0; f < StateTile::KFragments; ++f) {
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = StateTile::k_row(warp, f) + rc.row;
      hreg[f][s] = p.initial ? p.initial[(int64_t(bh) * Dim + row) * Dim + v0 + col + rc.col] : 0.0f;
    }
  }
  #pragma unroll 1
  for (int ct = 0; ct < p.shape.chunks(); ++ct) {
    int64_t const group = p.shape.group(b, h, ct);
    int const first = ct * Chunk;
    int const valid = FullChunks ? Chunk : min(Chunk, p.shape.sequence - first);
    Key::stage(sm.key, p.k + p.shape.input(b, first, qh, p.shape.q_heads), p.shape.q_heads * Dim, valid);
    Inverse::stage(sm.inverse, ws.inverse + inverse_base(group), Chunk, Chunk);
    Value::stage(sm.value, p.v + p.shape.input(b, first, h, p.shape.value_heads) + v0, p.shape.value_heads * Dim, valid);
    if (tid < Chunk) {
      float const prefix = ws.gates[group * Chunk + tid];
      sm.prefix_exp[tid] = expf(prefix);
      sm.relative_exp[tid] = expf(ws.gates[group * Chunk + valid - 1] - prefix);
      sm.beta[tid] = (FullChunks || tid < valid)
          ? float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h]) : 0.0f;
    }
    CUTE_UNROLL
    for (int f = 0; f < StateTile::KFragments; ++f) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.snapshot[Snapshot::offset(col + rc.col, StateTile::k_row(warp, f) + rc.row)] = BF16(hreg[f][s]);
      }
    }
    commit_wait();  // INPUTS_READY and BF16 H publication
    Snapshot::publish<StateTile::Threads>(sm.snapshot, ws.snapshots + snapshot_base(group) + v0 * Dim, Dim);

    int const row0 = StateTile::value_row(warp);
    float kh[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t a[4], bs[4];
      Key::load<true>(sm.key, row0, k, a);
      Snapshot::load<false>(sm.snapshot, col, k, bs);
      bf16_mma(kh, a, bs);
    }
    float row_decay[2], factor[2], beta[2];
    CUTE_UNROLL
    for (int half = 0; half < 2; ++half) {
      int const row = row0 + lane / 4 + half * 8;
      row_decay[half] = sm.relative_exp[row];
      factor[half] = sm.prefix_exp[row];
      beta[half] = sm.beta[row];
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = row0 + rc.row, v = col + rc.col, half = row_half(s);
      float const difference = float(sm.value[Value::offset(row, v)]) - factor[half] * kh[s];
      sm.residual[BIntermediate::offset(v, row)] = BF16((FullChunks || row < valid) ? beta[half] * difference : 0.0f);
    }
    __syncthreads();  // RESIDUAL_READY, input V readers have retired
    float value[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Chunk; k += 16) {
      uint32_t a[4], bs[4];
      Inverse::load<true>(sm.inverse, row0, k, a);
      BIntermediate::load<false>(sm.residual, col, k, bs);
      bf16_mma(value, a, bs);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = row0 + rc.row, v = col + rc.col;
      float const x = (FullChunks || row < valid) ? value[s] : 0.0f;
      sm.value[BIntermediate::offset(v, row)] = BF16(x);
      sm.scaled[BIntermediate::offset(v, row)] = BF16(x * row_decay[row_half(s)]);
    }
    __syncthreads();  // VALUES_READY: Vnew and scaledV are distinct BF16 casts
    BIntermediate::publish<StateTile::Threads>(sm.value, ws.vnew + value_base(group) + v0 * Chunk, Chunk);
    float const decay = sm.prefix_exp[valid - 1];
    CUTE_UNROLL
    for (int f = 0; f < StateTile::KFragments; ++f) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) hreg[f][s] *= decay;
    }
    CUTE_UNROLL
    for (int r = 0; r < Chunk; r += 16) {
      uint32_t bs[4];
      BIntermediate::load<false>(sm.scaled, col, r, bs);
      CUTE_UNROLL
      for (int f = 0; f < StateTile::KFragments; ++f) {
        uint32_t a[4];
        Key::load<true,true>(sm.key, r, StateTile::k_row(warp, f), a);
        bf16_mma(hreg[f], a, bs);
      }
    }
    __syncthreads();  // RETIRE before next chunk overwrites any shared operand
  }
  if (final) {
    CUTE_UNROLL
    for (int f = 0; f < StateTile::KFragments; ++f) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.final_h[(StateTile::k_row(warp, f) + rc.row) * ValueTile + col + rc.col] = hreg[f][s];
      }
    }
    __syncthreads();
    CUTE_UNROLL
    for (int it = 0; it < Dim * ValueTile / (4 * StateTile::Threads); ++it) {
      int const i = (tid + it * StateTile::Threads) * 4;
      *reinterpret_cast<float4*>(final + int64_t(bh) * Dim * Dim + (i / ValueTile) * Dim + v0 + i % ValueTile) =
          *reinterpret_cast<float4 const*>(sm.final_h + i);
    }
  }
}
template <bool FullChunks>
int launch_state(Inputs p, Workspace ws, float* final, hggcStream_t stream) {
  auto rc = hggcFuncSetAttribute(state_kernel<FullChunks>, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(StateStorage));
  if (rc) return int(rc);
  auto const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));
  state_kernel<FullChunks><<<grid, StateTile::Threads, sizeof(StateStorage), stream>>>(p, ws, final);
  return int(hggcGetLastError());
}
}  // namespace
int state(Inputs p, Workspace ws, float* final, hggcStream_t stream) {
  return p.shape.sequence % Chunk == 0 ? launch_state<true>(p, ws, final, stream)
                                      : launch_state<false>(p, ws, final, stream);
}
}  // namespace actlize_la::ppu15
