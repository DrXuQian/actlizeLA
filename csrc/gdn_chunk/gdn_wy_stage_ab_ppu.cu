// Independent opt-in address candidates. All admitted kernels remain in their
// original translation units; no copied mathematical model is used for the
// inverse. The shared prepare_inverse<Address> keeps the exact FP32 solve.
#include "gdn_wy_prepare.cuh"
#include "gdn_wy_stage_copy.cuh"
#include "gdn_qsa/ppu/wy_tiles.cuh"

namespace gdn_qsa::wy {

__global__ void __launch_bounds__(ParallelThreads)
gdn_wy_address_prepare(Inputs p, Workspace ws) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<PrepareStorage*>(storage);
  int const warp = unsigned(threadIdx.x) >> 5, lane = unsigned(threadIdx.x) & 31;
  int64_t const group = prepare_inverse<true>(p, ws, sm);
  BF16* inv = reinterpret_cast<BF16*>(sm.lower);
  #pragma unroll 1
  for (int tile = warp; tile < 32; tile += 4) {
    int const br = tile / 8, bc = tile % 8;
    float w[8] = {}, u[8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Chunk; k += 16) {
      uint32_t a[4], bk[4], bv[4];
      load<Chunk, Chunk>(inv, br * 16, k, a);
      load<Chunk, Dim, true>(sm.k, k, bc * 16, bk);
      load<Chunk, Dim, true>(sm.v, k, bc * 16, bv);
      bf16_mma(w, a, bk); bf16_mma(u, a, bv);
    }
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int64_t const at = tile_offset(group) + (br * 16 + rc.row) * Dim + bc * 16 + rc.col;
      ws.w[at] = BF16(w[s]); ws.u[at] = BF16(u[s]);
    }
  }
}

__global__ void __launch_bounds__(OutputTile::Threads)
gdn_wy_address_output(Inputs p, Workspace ws, BF16* output) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<TiledOutputStorage*>(storage);
  int const tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
  int const ct = int(blockIdx.x) % p.shape.chunks(), bh = int(blockIdx.x) / p.shape.chunks();
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads, qh = p.shape.q_head(h);
  int const first = ct * Chunk, valid = min(Chunk, p.shape.sequence - first);
  int64_t const group = p.shape.group(b, h, ct);
  state_stage<Chunk, Dim, OutputTile::Threads>(sm.q,
      p.q + p.shape.input(b, first, qh, p.shape.q_heads), int64_t(p.shape.q_heads) * Dim, valid);
  state_stage<Chunk, Dim, OutputTile::Threads>(sm.stage.k,
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
    state_stage<Dim, OutputTile::Panel, OutputTile::Threads>(sm.stage.h,
        ws.snapshots + state_offset(group) + panel, Dim, Dim);
    state_stage<Chunk, OutputTile::Panel, OutputTile::Threads>(sm.v,
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
    address_publish_tail<Chunk, OutputTile::Panel, OutputTile::Threads>(sm.stage.output,
        output + p.shape.input(b, first, h, p.shape.value_heads) + panel,
        int64_t(p.shape.value_heads) * Dim, valid);
    __syncthreads();  // exchange consumers retire before the next H prefetch
  }
}

int configure_stage_address(unsigned options) {
  if (!options || (options & ~(StageAddressOptions | PrepareRowsOptions)) || !valid_prepare_rows(options))
    return int(hggcErrorInvalidValue);
  auto const selection = stage_address_selection(options);
  if (selection.prepare) {
    auto const status = options & PrepareRowsOptions
        ? hggcError_t(configure_prepare_rows(options & PrepareRowsOptions))
        : hggcFuncSetAttribute(gdn_wy_address_prepare,
            hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(PrepareStorage));
    if (status != hggcSuccess) return int(status);
  }
  if (selection.output)
    return int(hggcFuncSetAttribute(gdn_wy_address_output,
        hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(TiledOutputStorage)));
  return int(hggcSuccess);
}

int launch_address_prepare(Inputs p, Workspace ws, gdn_arch::Stream stream) {
  gdn_wy_address_prepare<<<unsigned(p.shape.groups()), ParallelThreads, sizeof(PrepareStorage), stream>>>(p, ws);
  return int(hggcGetLastError());
}

int launch_address_output(Inputs p, Workspace ws, BF16* output, gdn_arch::Stream stream) {
  gdn_wy_address_output<<<unsigned(p.shape.groups()), OutputTile::Threads,
                          sizeof(TiledOutputStorage), stream>>>(p, ws, output);
  return int(hggcGetLastError());
}
}  // namespace gdn_qsa::wy
