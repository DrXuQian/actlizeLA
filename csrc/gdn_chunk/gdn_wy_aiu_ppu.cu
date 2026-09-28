// Matched AIU/SWZL delivery experiment. The admitted state/output bodies stay
// verbatim in their old TUs so the same binary retains an exact counterfactual.
// Arithmetic, warp ownership, workspace ABI and recurrence order are unchanged.
#include "gdn_wy_common.cuh"
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_tiles.cuh"
#include "gdn_qsa/ppu/wy_aiu.cuh"

namespace gdn_qsa::wy::aiu {

__global__ void __launch_bounds__(StateTile::Threads)
gdn_wy_aiu_state(Inputs p, Workspace ws, float* final) {
  using W = Tile<Chunk, Dim>;
  using H = Tile<Dim, ValueTile>;
  using V = Tile<Chunk, ValueTile>;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<TiledStateStorage*>(storage);
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
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
    W::stage(sm.w, ws.w + tile_offset(group), Dim, Chunk);
    W::stage(sm.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads),
             p.shape.q_heads * Dim, valid);
    if (tid < Chunk) sm.g[tid] = ws.gates[group * Chunk + tid];
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.snapshot[H::offset(StateTile::k_row(warp, k) + rc.row,
                               StateTile::column(warp) + rc.col)] = BF16(state[k][s]);
      }
    }
    commit_wait();
    H::publish<StateTile::Threads>(sm.snapshot, ws.snapshots + state_offset(group) + v0, Dim);
    V::stage(sm.u, ws.u + tile_offset(group) + v0, Dim, Chunk);
    cute::cp_async_fence();  // U overlaps W@H; H storage stays live
    float value[StateTile::ValueFragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t hs[4];
      H::load<true>(sm.snapshot, k, StateTile::column(warp), hs);
      CUTE_UNROLL
      for (int r = 0; r < StateTile::ValueFragments; ++r) {
        uint32_t w[4];
        W::load(sm.w, StateTile::value_row(warp, r), k, w);
        bf16_mma(value[r], w, hs);
      }
    }
    cute::cp_async_wait<0>();
    __syncthreads();
    float const last = sm.g[valid - 1];
    float row_gate[StateTile::ValueFragments][StateGateRows::Count];
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int half = 0; half < StateGateRows::Count; ++half) {
        int const row = StateTile::value_row(warp, r) + StateGateRows::row(lane, half);
        row_gate[r][half] = expf(last - sm.g[row]);
      }
    }
    CUTE_UNROLL
    for (int r = 0; r < StateTile::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = StateTile::value_row(warp, r) + rc.row;
        unsigned const at = V::offset(row, StateTile::column(warp) + rc.col);
        float const x = row < valid ? float(sm.u[at]) - value[r][s] : 0.0f;
        sm.u[at] = BF16(x);
        sm.scaled_v[at] = BF16(x * row_gate[r][StateGateRows::half(s)]);
      }
    }
    __syncthreads();
    V::publish<StateTile::Threads>(sm.u, ws.vnew + tile_offset(group) + v0, Dim);
    float const decay = expf(last);
    CUTE_UNROLL
    for (int k = 0; k < StateTile::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) state[k][s] *= decay;
    }
    CUTE_UNROLL
    for (int r = 0; r < Chunk; r += 16) {
      uint32_t value_operand[4];
      V::load<true>(sm.scaled_v, r, StateTile::column(warp), value_operand);
      CUTE_UNROLL
      for (int k = 0; k < StateTile::KFragments; ++k) {
        uint32_t key[4];
        W::load<true>(sm.k, r, StateTile::k_row(warp, k), key);
        bf16_mma(state[k], key, value_operand);
      }
    }
    __syncthreads();  // all readers retire before the next chunk's AIU writer
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
    state_publish_fp32<Dim, ValueTile, StateTile::Threads>(
        sm.final_h, final + int64_t(bh) * Dim * Dim + v0, Dim);
  }
}

__global__ void __launch_bounds__(OutputTile::Threads)
gdn_wy_aiu_output(Inputs p, Workspace ws, BF16* output) {
  using Q = Tile<Chunk, Dim>;
  using A = Tile<Chunk, Chunk>;
  using H = Tile<Dim, OutputTile::Panel>;
  using V = Tile<Chunk, OutputTile::Panel>;
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
    H::stage(sm.stage.h, ws.snapshots + state_offset(group) + panel, Dim, Dim);
    V::stage(sm.v, ws.vnew + tile_offset(group) + panel, Dim, Chunk);
    commit_wait();
    float acc[OutputTile::Fragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t q[4];
      Q::load(sm.q, OutputTile::row(warp), k, q);
      CUTE_UNROLL
      for (int c = 0; c < OutputTile::Fragments; ++c) {
        uint32_t hreg[4];
        H::load<true>(sm.stage.h, k, OutputTile::column(warp, c), hreg);
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
        V::load<true>(sm.v, k, OutputTile::column(warp, c), v);
        bf16_mma(acc[c], a, v);
      }
    }
    __syncthreads();  // the union's H readers finish before output exchange
    CUTE_UNROLL
    for (int c = 0; c < OutputTile::Fragments; ++c) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.stage.output[V::offset(OutputTile::row(warp) + rc.row,
                                  OutputTile::column(warp, c) + rc.col)] =
            BF16(acc[c][s] * 0.08838834764831845f);
      }
    }
    __syncthreads();
    V::publish<OutputTile::Threads>(sm.stage.output,
        output + p.shape.input(b, first, h, p.shape.value_heads) + panel,
        int64_t(p.shape.value_heads) * Dim, valid);
    __syncthreads();
  }
}

}  // namespace gdn_qsa::wy::aiu

namespace gdn_qsa::wy {
int configure_aiu_output() {
  return int(hggcFuncSetAttribute(aiu::gdn_wy_aiu_output,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledOutputStorage)));
}
int launch_aiu_output(Inputs p, Workspace ws, BF16* output, gdn_arch::Stream stream) {
  aiu::gdn_wy_aiu_output<<<unsigned(p.shape.groups()), OutputTile::Threads,
                          sizeof(TiledOutputStorage), stream>>>(p, ws, output);
  return int(hggcGetLastError());
}
int forward_aiu(Inputs p, Workspace ws, BF16* output, float* final,
                 gdn_arch::Stream stream, unsigned options) {
  // AIU descriptors have32-bit element pitches. Never silently truncate a
  // legal64-bit ordinary-copy stride, or silently fall back to the control.
  if (!aiu::Tile<Chunk, Dim>::admitted_stride(int64_t(p.shape.q_heads) * Dim) ||
      !aiu::Tile<Chunk, Dim>::admitted_stride(int64_t(p.shape.value_heads) * Dim))
    return int(hggcErrorInvalidValue);
  bool const split = bool(options & SplitPrepare);
  bool const pipeline = bool(options & StatePipeline);
  if (pipeline && options != (StatePipeline | SplitPrepare | AiuOptions)) return int(hggcErrorInvalidValue);
  if (split && (options & ~StatePipeline) != (SplitPrepare | AiuOptions)) return int(hggcErrorInvalidValue);
  return visit_aiu_options(options & AiuOptions, [&](auto use_state, auto use_output) {
    int rc = split ? configure_split_prepare() : configure_prepare_rows(PrepareRowsShared);
    if (rc) return rc;
    if constexpr (decltype(use_state)::value)
      rc = pipeline ? configure_state_pipeline() : int(hggcFuncSetAttribute(aiu::gdn_wy_aiu_state,
          hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledStateStorage)));
    else
      rc = configure_state_ab(StateOptions);
    if (rc) return rc;
    if constexpr (decltype(use_output)::value)
      rc = int(hggcFuncSetAttribute(aiu::gdn_wy_aiu_output,
          hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledOutputStorage)));
    else
      rc = configure_tiled(32u);
    if (rc) return rc;
    rc = split ? launch_split_prepare(p, ws, stream)
               : launch_prepare_rows(p, ws, stream, PrepareRowsShared);
    if (rc) return rc;
    if constexpr (decltype(use_state)::value) {
      if (pipeline) {
        rc = launch_state_pipeline(p, ws, final, stream);
      } else {
        unsigned const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));
        aiu::gdn_wy_aiu_state<<<grid, StateTile::Threads, sizeof(TiledStateStorage), stream>>>(p, ws, final);
        rc = int(hggcGetLastError());
      }
    } else {
      rc = launch_state_ab(p, ws, final, stream, StateOptions);
    }
    if (rc) return rc;
    if constexpr (decltype(use_output)::value) {
      aiu::gdn_wy_aiu_output<<<unsigned(p.shape.groups()), OutputTile::Threads,
                              sizeof(TiledOutputStorage), stream>>>(p, ws, output);
      return int(hggcGetLastError());
    } else {
      return launch_tiled_output(p, ws, output, stream);
    }
  }, int(hggcErrorInvalidValue));
}
}  // namespace gdn_qsa::wy
