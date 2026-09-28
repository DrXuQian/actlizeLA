// Compute-tile alternatives; no original-route or arithmetic-precision changes.
#include "gdn_wy_prepare.cuh"
#include "gdn_qsa/ppu/wy_tiles.cuh"

namespace gdn_qsa::wy {

__global__ void __launch_bounds__(PrepareTile::Threads)
gdn_wy_tiled_prepare(Inputs p, Workspace ws) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<PrepareStorage*>(storage);
  int const tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int64_t const group = prepare_inverse(p, ws, sm);
  BF16 const* inv = reinterpret_cast<BF16 const*>(sm.lower);
  // The FP32 inverse is dead after prepare_inverse publishes the BF16 inverse.
  // Retain that BF16 copy across panels; exchange W/U in the dead FP32 array.
  BF16* packed_w = reinterpret_cast<BF16*>(sm.inverse);
  BF16* packed_u = packed_w + Chunk * PrepareTile::Panel;
  static_assert(2 * Chunk * PrepareTile::Panel * sizeof(BF16) == sizeof(sm.inverse));
  #pragma unroll 1
  for (int panel = 0; panel < Dim; panel += PrepareTile::Panel) {
    float w[PrepareTile::Fragments][8] = {}, u[PrepareTile::Fragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Chunk; k += 16) {
      uint32_t a[4];
      load<Chunk, Chunk>(inv, PrepareTile::row(warp), k, a);
      CUTE_UNROLL
      for (int n = 0; n < PrepareTile::Fragments; ++n) {
        uint32_t bk[4], bv[4];
        int const col = panel + PrepareTile::column(n);
        load<Chunk, Dim, true>(sm.k, k, col, bk);
        load<Chunk, Dim, true>(sm.v, k, col, bv);
        bf16_mma(w[n], a, bk);
        bf16_mma(u[n], a, bv);
      }
    }
    CUTE_UNROLL
    for (int n = 0; n < PrepareTile::Fragments; ++n) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = PrepareTile::row(warp) + rc.row;
        int const col = PrepareTile::column(n) + rc.col;
        int const at = swizzle<Chunk, PrepareTile::Panel>(row, col);
        packed_w[at] = BF16(w[n][s]);
        packed_u[at] = BF16(u[n][s]);
      }
    }
    __syncthreads();
    publish_bf16<Chunk, PrepareTile::Panel, PrepareTile::Threads>(
        packed_w, ws.w + tile_offset(group) + panel, Dim, tid);
    publish_bf16<Chunk, PrepareTile::Panel, PrepareTile::Threads>(
        packed_u, ws.u + tile_offset(group) + panel, Dim, tid);
    __syncthreads();  // all exchange readers retire before the next panel
  }
}

__global__ void __launch_bounds__(StateTile::Threads)
gdn_wy_tiled_state(Inputs p, Workspace ws, float* final) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<TiledStateStorage*>(storage);
  int const tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
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
    int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
    stage<Chunk, Dim, StateTile::Threads>(sm.w, ws.w + tile_offset(group), Dim, Chunk);
    stage<Chunk, Dim, StateTile::Threads>(sm.k,
        p.k + p.shape.input(b, first, qh, p.shape.q_heads), int64_t(p.shape.q_heads) * Dim, valid);
    if (tid < Chunk) sm.g[tid] = ws.gates[group * Chunk + tid];
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.snapshot[swizzle<Dim, ValueTile>(StateTile::k_row(warp, k) + rc.row,
            StateTile::column(warp) + rc.col)] = BF16(state[k][s]);
      }
    }
    commit_wait();  // W/K and all four warps' H fragments become visible
    publish_bf16<Dim, ValueTile, StateTile::Threads>(sm.snapshot,
        ws.snapshots + state_offset(group) + v0, Dim, tid);
    stage<Chunk, ValueTile, StateTile::Threads>(sm.u,
        ws.u + tile_offset(group) + v0, Dim, Chunk);
    cute::cp_async_fence();  // U delivery overlaps W@H, without destroying H
    float value[StateTile::ValueFragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t hs[4];
      load<Dim, ValueTile, true>(sm.snapshot, k, StateTile::column(warp), hs);
      CUTE_UNROLL
      for (int r = 0; r < StateTile::ValueFragments; ++r) {
        uint32_t w[4];
        load<Chunk, Dim>(sm.w, StateTile::value_row(warp, r), k, w);
        bf16_mma(value[r], w, hs);
      }
    }
    cute::cp_async_wait<0>();
    __syncthreads();
    float const last = sm.g[valid - 1];
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = StateTile::value_row(warp, r) + rc.row;
        int const at = swizzle<Chunk, ValueTile>(row, StateTile::column(warp) + rc.col);
        float const x = row < valid ? float(sm.u[at]) - value[r][s] : 0.0f;
        sm.u[at] = BF16(x);
        sm.scaled_v[at] = BF16(x * expf(last - sm.g[row]));
      }
    }
    __syncthreads();  // both row halves feed every warp's K^T V update
    publish_bf16<Chunk, ValueTile, StateTile::Threads>(sm.u,
        ws.vnew + tile_offset(group) + v0, Dim, tid);
    float const decay = expf(last);
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) state[k][s] *= decay;
    }
    // Interchange independent output K tiles, NOT reduction terms. Each
    // state's contributions still arrive in r=0,16,32,48 order.
    CUTE_UNROLL
    for (int r = 0; r < Chunk; r += 16) {
      uint32_t value_operand[4];
      load<Chunk, ValueTile, true>(sm.scaled_v, r, StateTile::column(warp), value_operand);
      CUTE_UNROLL
      for (int k = 0; k < StateTile::KFragments; ++k) {
        uint32_t key[4];
        load<Chunk, Dim, true>(sm.k, r, StateTile::k_row(warp, k), key);
        bf16_mma(state[k], key, value_operand);
      }
    }
    __syncthreads();  // finish shared consumers before the next chunk
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
    publish_fp32<Dim, ValueTile, StateTile::Threads>(sm.final_h,
        final + int64_t(bh) * Dim * Dim + v0, Dim, tid);
  }
}

__global__ void __launch_bounds__(OutputTile::Threads)
gdn_wy_tiled_output(Inputs p, Workspace ws, BF16* output) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<TiledOutputStorage*>(storage);
  int const tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks(), bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads, qh = p.shape.q_head(h);
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int64_t const group = p.shape.group(b, h, ct);
  stage<Chunk, Dim, OutputTile::Threads>(sm.q,
      p.q + p.shape.input(b, first, qh, p.shape.q_heads), int64_t(p.shape.q_heads) * Dim, valid);
  stage<Chunk, Dim, OutputTile::Threads>(sm.stage.k,
      p.k + p.shape.input(b, first, qh, p.shape.q_heads), int64_t(p.shape.q_heads) * Dim, valid);
  if (tid < Chunk) sm.g[tid] = ws.gates[group * Chunk + tid];
  commit_wait();
  float attention[OutputTile::Fragments][8] = {};
  CUTE_UNROLL
  for (int k = 0; k < Dim; k += 16) {
    uint32_t q[4];
    load<Chunk, Dim>(sm.q, OutputTile::row(warp), k, q);
    CUTE_UNROLL
    for (int c = 0; c < OutputTile::Fragments; ++c) {
      uint32_t key[4];
      load<Chunk, Dim>(sm.stage.k, OutputTile::column(warp, c), k, key);
      bf16_mma(attention[c], q, key);
    }
  }
  CUTE_UNROLL
  for (int c = 0; c < OutputTile::Fragments; ++c) {
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = OutputTile::row(warp) + rc.row, col = OutputTile::column(warp, c) + rc.col;
      sm.attn[swizzle<Chunk, Chunk>(row, col)] = BF16(
          row >= col && row < valid ? attention[c][s] * expf(sm.g[row] - sm.g[col]) : 0.0f);
    }
  }
  __syncthreads();  // K is dead before stage becomes H
  // The two rows in a native accumulator share their gate across all panels.
  float row_gate[2];
  CUTE_UNROLL
  for (int row = 0; row < 2; ++row)
    row_gate[row] = expf(sm.g[OutputTile::row(warp) + lane / 4 + row * 8]);
  #pragma unroll 1
  for (int panel = 0; panel < Dim; panel += OutputTile::Panel) {
    stage<Dim, OutputTile::Panel, OutputTile::Threads>(sm.stage.h,
        ws.snapshots + state_offset(group) + panel, Dim, Dim);
    stage<Chunk, OutputTile::Panel, OutputTile::Threads>(sm.v,
        ws.vnew + tile_offset(group) + panel, Dim, Chunk);
    commit_wait();
    float acc[OutputTile::Fragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t q[4];
      load<Chunk, Dim>(sm.q, OutputTile::row(warp), k, q);
      CUTE_UNROLL
      for (int c = 0; c < OutputTile::Fragments; ++c) {
        uint32_t hreg[4];
        load<Dim, OutputTile::Panel, true>(sm.stage.h, k, OutputTile::column(warp, c), hreg);
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
      load<Chunk, Chunk>(sm.attn, OutputTile::row(warp), k, a);
      CUTE_UNROLL
      for (int c = 0; c < OutputTile::Fragments; ++c) {
        uint32_t v[4];
        load<Chunk, OutputTile::Panel, true>(sm.v, k, OutputTile::column(warp, c), v);
        bf16_mma(acc[c], a, v);
      }
    }
    __syncthreads();  // every H reader retires before the alias becomes output
    CUTE_UNROLL
    for (int c = 0; c < OutputTile::Fragments; ++c) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const at = swizzle<Chunk, OutputTile::Panel>(OutputTile::row(warp) + rc.row,
            OutputTile::column(warp, c) + rc.col);
        sm.stage.output[at] = BF16(acc[c][s] * 0.08838834764831845f);
      }
    }
    __syncthreads();
    publish_bf16<Chunk, OutputTile::Panel, OutputTile::Threads>(sm.stage.output,
        output + p.shape.input(b, first, h, p.shape.value_heads) + panel,
        int64_t(p.shape.value_heads) * Dim, tid, valid);
    __syncthreads();  // exchange consumers retire before the next H prefetch
  }
}

int configure_tiled(unsigned delivery) {
  hggcError_t status = hggcSuccess;
  if (delivery & 8) {
    status = hggcFuncSetAttribute(gdn_wy_tiled_prepare,
        hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(PrepareStorage));
    if (status != hggcSuccess) return int(status);
  }
  if (delivery & 16) {
    status = delivery & StateOptions ? hggcError_t(configure_state_ab(delivery & StateOptions)) :
        hggcFuncSetAttribute(gdn_wy_tiled_state,
            hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledStateStorage));
    if (status != hggcSuccess) return int(status);
  }
  if ((delivery & 32) && !(delivery & OutputAddress))
    status = hggcFuncSetAttribute(gdn_wy_tiled_output,
        hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledOutputStorage));
  return int(status);
}
int launch_tiled_prepare(Inputs p, Workspace ws, gdn_arch::Stream stream) {
  gdn_wy_tiled_prepare<<<unsigned(p.shape.groups()), PrepareTile::Threads, sizeof(PrepareStorage), stream>>>(p, ws);
  return int(hggcGetLastError());
}
int launch_tiled_state(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream, unsigned state_options) {
  if (state_options) return launch_state_ab(p, ws, final, stream, state_options);
  unsigned const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));
  gdn_wy_tiled_state<<<grid, StateTile::Threads, sizeof(TiledStateStorage), stream>>>(p, ws, final);
  return int(hggcGetLastError());
}
int launch_tiled_output(Inputs p, Workspace ws, BF16* output, gdn_arch::Stream stream) {
  gdn_wy_tiled_output<<<unsigned(p.shape.groups()), OutputTile::Threads, sizeof(TiledOutputStorage), stream>>>(p, ws, output);
  return int(hggcGetLastError());
}
}  // namespace gdn_qsa::wy
