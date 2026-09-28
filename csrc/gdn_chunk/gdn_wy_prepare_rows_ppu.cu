// Row-factor-only ablation on the admitted address-prepare kernel.
// State/output and all existing prepare bodies remain frozen controls.
#include "gdn_wy_prepare.cuh"

namespace gdn_qsa::wy {

template <int RowCache>
__global__ void __launch_bounds__(ParallelThreads)
gdn_wy_rows_prepare(Inputs p, Workspace ws) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<PrepareStorage*>(storage);
  int const warp = unsigned(threadIdx.x) >> 5, lane = unsigned(threadIdx.x) & 31;
  int64_t const group = prepare_inverse<true, RowCache>(p, ws, sm);
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

int configure_prepare_rows(unsigned options) {
  return visit_prepare_rows(options, [&](auto mode) {
    return int(hggcFuncSetAttribute(gdn_wy_rows_prepare<decltype(mode)::value>,
        hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(PrepareStorage)));
  }, int(hggcErrorInvalidValue));
}

int launch_prepare_rows(Inputs p, Workspace ws, gdn_arch::Stream stream, unsigned options) {
  return visit_prepare_rows(options, [&](auto mode) {
    gdn_wy_rows_prepare<decltype(mode)::value>
        <<<unsigned(p.shape.groups()), ParallelThreads, sizeof(PrepareStorage), stream>>>(p, ws);
    return int(hggcGetLastError());
  }, int(hggcErrorInvalidValue));
}
}  // namespace gdn_qsa::wy
