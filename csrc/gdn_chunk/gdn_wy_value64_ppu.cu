// 最终V64/256 state：保留C64残差数学、BF16边界与FP32 carried state。
#include "gdn_wy_common.cuh"
#include "gdn_wy_state_copy.cuh"
#include "gdn_qsa/ppu/wy_value64.cuh"

namespace gdn_qsa::wy::value64 {
template <unsigned TileWidth = 64, unsigned ThreadCount = 256>
__global__ void __launch_bounds__(ThreadCount)
gdn_wy_value64_state(Inputs p, Workspace ws, float* final) {
  constexpr unsigned Width = TileWidth;
  using Geometry = GeometryT<Width, ThreadCount>;
  using Storage = StorageT<Width>;
  using Snapshot = SnapshotT<Width, ThreadCount>;
  using PublishedValue = PublishedValueT<Width, ThreadCount>;
  using BIntermediate = MicroB<Chunk, Width, ThreadCount>;
  using Value = aiu::Tile<Chunk, Width>;
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<Storage*>(storage);
  auto& data = sm.math;
  auto& coefficients = sm.coefficients;
  unsigned const tid = unsigned(threadIdx.x), warp = tid / 32, lane = tid % 32;
  unsigned const h_base = Snapshot::producer_base(warp, lane);
  unsigned const b_base = BIntermediate::producer_base(warp, lane);
  int const slice = int(blockIdx.x) % (Dim / Width), bh = int(blockIdx.x) / (Dim / Width);
  int const h = bh % p.shape.value_heads, b = bh / p.shape.value_heads;
  int const v0 = slice * Width, qh = p.shape.q_head(h);
  float state[Geometry::KFragments][8];
  CUTE_UNROLL
  for (int k = 0; k < Geometry::KFragments; ++k) {
    CUTE_UNROLL
    for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      int const row = Geometry::k_row(warp, k) + rc.row;
      int const col = v0 + Geometry::column(warp) + rc.col;
      state[k][s] = p.initial ? p.initial[(int64_t(bh) * Dim + row) * Dim + col] : 0.0f;
    }
  }
  #pragma unroll 1
  for (int ct = 0; ct < p.shape.chunks(); ++ct) {
    int64_t const group = p.shape.group(b, h, ct);
    int const first = ct * Chunk, valid = Chunk;
    Key::stage(sm.k, p.k + p.shape.input(b, first, qh, p.shape.q_heads), p.shape.q_heads * Dim, valid);
    Inverse::stage(sm.inverse, ws.w + state_offset(group), Chunk, Chunk);
    Value::stage(data.value, p.v + p.shape.input(b, first, h, p.shape.value_heads) + v0,
                 p.shape.value_heads * Dim, valid);
    if (tid < Chunk) {
      coefficients.publish(tid, ws.gates[group * Chunk + tid], ws.gates[group * Chunk + valid - 1]);
      sm.beta[tid] = float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h]);
    }
    CUTE_UNROLL
    for (int k = 0; k < Geometry::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s)
        data.snapshot[Snapshot::producer_offset(h_base, k, s)] = BF16(state[k][s]);
    }
    commit_wait();
    Snapshot::publish(data.snapshot, ws.snapshots + Snapshot::workspace_offset(group, 0, v0));
    float kh[Geometry::ValueFragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Dim; k += 16) {
      uint32_t hs[4];
      Snapshot::load(data.snapshot, k, Geometry::column(warp), hs);
      CUTE_UNROLL
      for (int r = 0; r < Geometry::ValueFragments; ++r) {
        uint32_t key[4];
        Key::load(sm.k, Geometry::value_row(warp, r), k, key);
        bf16_mma(kh[r], key, hs);
      }
    }
    float row_decay[Geometry::ValueFragments][StateGateRows::Count];
    CUTE_UNROLL
    for (int r = 0; r < Geometry::ValueFragments; ++r) {
      float factor[StateGateRows::Count], beta[StateGateRows::Count];
      CUTE_UNROLL
      for (int half = 0; half < StateGateRows::Count; ++half) {
        int const row = Geometry::value_row(warp, r) + StateGateRows::row(lane, half);
        factor[half] = coefficients.prefix[row];
        beta[half] = sm.beta[row];
        row_decay[r][half] = coefficients.relative[row];
      }
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        int const row = Geometry::value_row(warp, r) + rc.row;
        unsigned const at = Value::offset(row, Geometry::column(warp) + rc.col);
        int const half = StateGateRows::half(s);
        float const difference = float(data.value[at]) - factor[half] * kh[r][s];
        data.residual[BIntermediate::producer_offset(b_base, r, s)] = BF16(beta[half] * difference);
      }
    }
    __syncthreads();
    float value[Geometry::ValueFragments][8] = {};
    CUTE_UNROLL
    for (int k = 0; k < Chunk; k += 16) {
      uint32_t residual[4];
      BIntermediate::load(data.residual, k, Geometry::column(warp), residual);
      CUTE_UNROLL
      for (int r = 0; r < Geometry::ValueFragments; ++r) {
        uint32_t inverse[4];
        Inverse::load(sm.inverse, Geometry::value_row(warp, r), k, inverse);
        bf16_mma(value[r], inverse, residual);
      }
    }
    CUTE_UNROLL
    for (int r = 0; r < Geometry::ValueFragments; ++r) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        float const x = value[r][s];
        data.value[PublishedValue::producer_offset(b_base, r, s)] = BF16(x);
        sm.scaled[BIntermediate::producer_offset(b_base, r, s)] = BF16(x * row_decay[r][StateGateRows::half(s)]);
      }
    }
    __syncthreads();
    PublishedValue::publish(data.value, ws.vnew + PublishedValue::workspace_offset(group, 0, v0));
    float const decay = coefficients.prefix[valid - 1];
    CUTE_UNROLL
    for (int k = 0; k < Geometry::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) state[k][s] *= decay;
    }
    CUTE_UNROLL
    for (int r = 0; r < Chunk; r += 16) {
      uint32_t scaled[4];
      BIntermediate::load(sm.scaled, r, Geometry::column(warp), scaled);
      CUTE_UNROLL
      for (int k = 0; k < Geometry::KFragments; ++k) {
        uint32_t key[4];
        Key::load<true>(sm.k, r, Geometry::k_row(warp, k), key);
        bf16_mma(state[k], key, scaled);
      }
    }
    __syncthreads();
  }
  // 所有读者已退休；32KiB union改用FP32 final_h，不降精度。
  if (final) {
    CUTE_UNROLL
    for (int k = 0; k < Geometry::KFragments; ++k) {
      CUTE_UNROLL
      for (int s = 0; s < 8; ++s) {
        auto const rc = result_coord(lane, s);
        sm.final_h[(Geometry::k_row(warp, k) + rc.row) * Width + Geometry::column(warp) + rc.col] = state[k][s];
      }
    }
    __syncthreads();
    state_publish_fp32<Dim, Width, Geometry::Threads>(sm.final_h, final + int64_t(bh) * Dim * Dim + v0, Dim);
  }
}
int configure_value64_warps8() {
  return int(hggcFuncSetAttribute(gdn_wy_value64_state<64, 256>,
      hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(StorageT<64>)));
}
int launch_value64_warps8(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream) {
  gdn_wy_value64_state<64, 256><<<unsigned(int64_t(p.shape.batch) * p.shape.value_heads * 2),
      256, sizeof(StorageT<64>), stream>>>(p, ws, final);
  return int(hggcGetLastError());
}
}  // namespace gdn_qsa::wy::value64
