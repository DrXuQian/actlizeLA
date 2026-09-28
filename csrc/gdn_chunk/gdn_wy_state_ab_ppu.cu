// Independent state-only experiments. The admitted tiled-state control stays
// verbatim in gdn_wy_tiles_ppu.cu: factoring it through a device helper changed
// its generated instructions even with both options off. Do not re-deduplicate
// that frozen counterfactual without re-establishing the machine-code baseline.
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_tiles.cuh"

namespace gdn_qsa::wy {

template <bool Address, bool RowReuse>
__global__ void __launch_bounds__(StateTile::Threads)
gdn_wy_state_ab(Inputs p, Workspace ws, float* final) {
  static_assert(Address || RowReuse, "use the unchanged tiled-state kernel for the control");
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
    if constexpr (Address) {
      state_stage<Chunk, Dim, StateTile::Threads>(sm.w, ws.w + tile_offset(group), Dim, Chunk);
      state_stage<Chunk, Dim, StateTile::Threads>(sm.k,
          p.k + p.shape.input(b, first, qh, p.shape.q_heads), int64_t(p.shape.q_heads) * Dim, valid);
    } else {
      stage<Chunk, Dim, StateTile::Threads>(sm.w, ws.w + tile_offset(group), Dim, Chunk);
      stage<Chunk, Dim, StateTile::Threads>(sm.k,
          p.k + p.shape.input(b, first, qh, p.shape.q_heads), int64_t(p.shape.q_heads) * Dim, valid);
    }
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
    if constexpr (Address) {
      state_publish_bf16<Dim, ValueTile, StateTile::Threads>(sm.snapshot,
          ws.snapshots + state_offset(group) + v0, Dim);
      state_stage<Chunk, ValueTile, StateTile::Threads>(sm.u,
          ws.u + tile_offset(group) + v0, Dim, Chunk);
    } else {
      publish_bf16<Dim, ValueTile, StateTile::Threads>(sm.snapshot,
          ws.snapshots + state_offset(group) + v0, Dim, tid);
      stage<Chunk, ValueTile, StateTile::Threads>(sm.u,
          ws.u + tile_offset(group) + v0, Dim, Chunk);
    }
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
    float row_gate[StateTile::ValueFragments][StateGateRows::Count];
    if constexpr (RowReuse) {
      CUTE_UNROLL
      for (int r = 0; r < StateTile::ValueFragments; ++r) {
        CUTE_UNROLL
        for (int half = 0; half < StateGateRows::Count; ++half) {
          int const row = StateTile::value_row(warp, r) + StateGateRows::row(lane, half);
          row_gate[r][half] = expf(last - sm.g[row]);
        }
      }
    }
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = StateTile::value_row(warp, r) + rc.row;
        int const at = swizzle<Chunk, ValueTile>(row, StateTile::column(warp) + rc.col);
        float const x = row < valid ? float(sm.u[at]) - value[r][s] : 0.0f;
        sm.u[at] = BF16(x);
        if constexpr (RowReuse)
          sm.scaled_v[at] = BF16(x * row_gate[r][StateGateRows::half(s)]);
        else
          sm.scaled_v[at] = BF16(x * expf(last - sm.g[row]));
      }
    }
    __syncthreads();  // both row halves feed every warp's K^T V update
    if constexpr (Address)
      state_publish_bf16<Chunk, ValueTile, StateTile::Threads>(sm.u,
          ws.vnew + tile_offset(group) + v0, Dim);
    else
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
    if constexpr (Address)
      state_publish_fp32<Dim, ValueTile, StateTile::Threads>(sm.final_h,
          final + int64_t(bh) * Dim * Dim + v0, Dim);
    else
      publish_fp32<Dim, ValueTile, StateTile::Threads>(sm.final_h,
          final + int64_t(bh) * Dim * Dim + v0, Dim, tid);
  }
}

int configure_state_ab(unsigned options) {
  return visit_state_options(options, [&](auto address, auto gates) {
    return int(hggcFuncSetAttribute(gdn_wy_state_ab<decltype(address)::value, decltype(gates)::value>,
        hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledStateStorage)));
  }, int(hggcErrorInvalidValue));
}

int launch_state_ab(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream, unsigned options) {
  unsigned const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));
  return visit_state_options(options, [&](auto address, auto gates) {
    gdn_wy_state_ab<decltype(address)::value, decltype(gates)::value>
        <<<grid, StateTile::Threads, sizeof(TiledStateStorage), stream>>>(p, ws, final);
    return int(hggcGetLastError());
  }, int(hggcErrorInvalidValue));
}
}  // namespace gdn_qsa::wy
